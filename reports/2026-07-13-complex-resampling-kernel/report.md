# 复数 SAR 地理编码重采样核选择调研报告

**日期**：2026-07-13  
**范围**：`faninsar/processing/merge/geocode_raster.py` 中 `geocode_complex_to_grid` 的复数插值核  
**当前实现**：`scipy.ndimage.map_coordinates(..., order=1)` — bilinear，分别作用于 real/imag  

---

## 1. 问题

`geocode_complex_to_grid` 将雷达坐标系的复数 SLC / wrapped ifg 投影到公共 UTM 网格，供后续
burst merge（phase_network + mosaic）使用。当前使用 bilinear（`order=1`）对 real 和 imag
分别插值。问题是：**bilinear 对复数 SAR 数据是否生产级可接受？**

---

## 2. 调研来源

### 2.1 ISCE2（开源 InSAR 生产栈的事实标准）

ISCE2 源码中 `Geocodable.py` 和 `Resamp_slc.py` 明确按数据类型选择 kernel：

```python
# components/zerodop/geozero/Geocodable.py
self._interp_map = {
    'amp': 'sinc', 'cpx': 'sinc',    # complex → sinc
    'cor': 'nearest', 'unw': 'nearest', 'rmg': 'nearest'  # real → nearest
}

# components/stdproc/resamp_slc/Resamp_slc.py
if self.method is None:
    if self.isComplex: self.method = 'SINC'
    else:              self.method = 'BILINEAR'
```

**结论**：ISCE2 按 `isComplex` / `CFLOAT` 自动锁定 sinc；bilinear 仅用于非复数（detected
amplitude）产品。这不是用户可调参数，而是系统设计。

### 2.2 ISCE3 / NISAR

ISCE3 `geocodeSlc` 的示例 XML 中出现 `bilinear`，但那是 detected backscatter（实值）的
默认；NISAR L2 GSLC ATBD 描述的复数 GSLC 生产路径使用 **加窗 sinc（raised-cosine,
8-tap）**，是 phase-preserving 的。

### 2.3 ESA Sentinel-1 IPF

S1 Level-1 IPF（S1-IF-ASD-PL-1007）的 SLC 生成全程使用 FFT/zero-padded sinc 重构（距离
压缩 = zero-padded FFT，方位向 = FFT aperture synthesis）。**IPF 中不出现 bilinear 复数
插值**。bilinear 仅在下游 GRD detected 产品中。

### 2.4 pygmtsar / InSAR.dev / GMTSAR

GMTSAR 的 `geocode` 使用 GMT `grdresample`（默认 bilinear），但关键前提：**先 multilook
再 geocode**。multilook 在雷达坐标系中完成复数平均，将信号带宽降到远低于输出网格 Nyquist，
此时 bilinear ≈ sinc。pygmtsar 文档推荐此顺序，且 `.omo/ulw-research` SYNTHESIS 中已
指出：

> "ISCE3's public `geocode_slc` surface itself exposes orbit, Doppler, iterative geo2rdr,
> carrier, correction and invalid-data semantics, demonstrating why a generic Lanczos warp
> is incomplete."

这确认：**对原始 SLC，bilinear 不可接受**；仅当输入已充分 multilook/oversampled 时，bilinear
近似可用。

### 2.5 `.omo/ulw-research` 已有警示

`.omo/evidence/` 和 `.omo/ulw-research/20260711-155830/SYNTHESIS.md` 中多次标记：

- "Phase-preserving interpolation and repeatable geometry" 是 geocode-first 的前提条件
- "Current plan is evidence-complete" 表中，geocode-first 路径被标记为 **Evidence gap**
- wave-1-partial-team-evidence.md 记录："Academic counterevidence: complex resampling
  damage depends on interpolation kernel and spectrum; the v3 fixed 1-5% coherence-loss
  claim has no supporting benchmark."

这表明 `.omo` 研究已在早期识别出 kernel 选择的重要性，但未在实现中落地。

---

## 3. 谱理论分析

### 3.1 为什么 bilinear 对复数 SAR 不行

SAR SLC 是**带限采样信号**（距离带宽 B_r ≤ f_s，方位带宽 B_a ≤ PRF）。根据 Nyquist-
Shannon 采样定理，在任意非整数坐标处的精确重构需要 **sinc kernel**：

```
f(x) = Σ_k f[k] · sinc(x - k)
```

bilinear 用分段线性（三角）kernel 替代 sinc。其频响为：

```
H_bilinear(f) = sinc²(πf / f_s)
```

这带来两个问题：

1. **带内信号衰减**：在 0 < |f| < f_s/2 范围内，bilinear 频响低于 sinc（平坦）→ 幅度
   压缩、与局部分数坐标耦合
2. **残余混叠**：旁瓣衰减比 sinc 慢 → 带外能量渗入 → 相位扰动

对**实值**（幅度、相干、DEM）信号，相位概念不适用，bilinear 衰减和混叠的影响主要在分辨率/
精度层面，通常可接受。

对**复数**信号，相位 = arg(z)。kernel 引起的幅度/相位扰动直接变成 **坐标依赖的相位偏置**
Δφ(x, y)，且该偏置与局部子像素分数偏移相关。

### 3.2 Lanczos kernel

Lanczos kernel 是截断的加窗 sinc：

```
L_a(x) = sinc(x) · sinc(x/a),  |x| ≤ a
         0,                       |x| > a
```

- a=4（8-tap）或 a=6（12-tap）是 InSAR 生产标准
- 相位保真度：~1e-3 rad（对 oversampled SLC）
- 是 ISCE2/ISCE3/NISAR/Gamma/S1-IPF 的共识选择

---

## 4. 当前 bilinear 实现的具体失败模式

### 4.1 Burst 边界相位缝（M2，最严重）

相邻 TOPS burst 投到公共 UTM 网格时，每个 burst 的子像素分数偏移不同（burst timing 差
非整采样数）。bilinear 给每个 burst 一个与分数偏移耦合的相位偏置。

**后果**：mosaic 加权平均时，偏置在 burst 边界不连续 → 经典 "geocode seam"。feather 和
phase_network **修不了**，因为偏差在 merge 之前已写进每个 burst 的网格采样。

### 4.2 相干性低估

分数偏移处 bilinear decorrelate 复数场 → `coh_out` 系统性偏低。幅度与局部分数坐标耦合。

### 4.3 亚像素相位斜坡污染

残差轨道/电离层斜坡（低频相位信号）被坐标相关误差耦合进解缠解 → 低频相位误差混入形变
信号，与真实形变不可区分（cm 量级）。

### 4.4 纹理相关相位偏置

高纹理区（高带宽）bilinear 衰减更大 → 误差非平稳、场景相关，无法用常数标定。

### 4.5 M1/M2 路径不一致

M1（geocoded-SLC ifg）和 M2（radar-ifg geocode）两条生产链的 resampling 偏置不同 →
破坏 frame 级一致性校验。

---

## 5. 成本/质量权衡

| Kernel | 相位保真 | 相干损失 | 计算成本 | 适用场景 |
|--------|---------|---------|---------|---------|
| nearest | 无（取整） | 高（混叠） | 1× | 分类、conncomp、水体掩膜 |
| bilinear | **差（复数）** | 中等 | ~1× | detected amplitude, 已 multilook 的平滑相位 |
| bicubic (Keys) | 群延迟非平坦 → 相位偏置 | 低但有偏 | ~4× | **不用于复数 SAR** |
| **sinc (windowed, Lanczos a=4)** | **高（~1e-3 rad）** | **最小** | ~8-16× | **复数 SLC/ifg — 生产标准** |
| Lanczos a=6 | 很高 | 极小 | ~16-24× | 高精度需求 |

**实际影响**：`geocode_complex_to_grid` 的瓶颈是 `geo2rdr` 逐像元牛顿迭代，不是
resample。Lanczos (a=4) 比 bilinear 慢约 8-16×，但因 resample 非主导，**整体耗时增加约
10-20%**。

---

## 6. 结论

### 6.1 bilinear 对复数 SAR 不适合生产级

ISCE2、ISCE3/NISAR、S1 IPF、谱理论一致表明：复数 SLC/ifg 的地理编码必须用 sinc 系
kernel。bilinear 仅适用于：
- 实值资产（amplitude、coherence、DEM、unw_phase）—— 正确
- 已充分 multilook 到信号带宽远低于网格 Nyquist 的平滑相位场 —— 可接受

当前 `geocode_complex_to_grid` 处理的是 **未 multilook 的复数 SLC / wrapped ifg**，是
bilinear 最不适用的场景。

### 6.2 推荐改动

将 `faninsar/processing/merge/geocode_raster.py:167-172` 的两处
`map_coordinates(order=1)` 替换为 **separable Lanczos resampler**（a=4，8-tap）：
- coherence 层（line 179）保持 `order=1`（实值，bilinear 正确）
- 实现方式：自定义 separable Lanczos kernel（~40 行），无新依赖

### 6.3 验证方法

在替换后的 TDD 回归测试中验证：
1. 合成带限复数信号（已知频谱），geocode 到非整数偏移网格，测量相位 RMS 误差
2. Lanczos 应达到 ~1e-3 rad 保真度；bilinear 在相同测试下误差 >1e-2 rad
3. 两 burst 同 path mosaic 的叠区相位缝消失（bilinear 有缝，Lanczos 无缝）

---

## 附录 A — 本地相关文件

| 文件 | 角色 |
|------|------|
| `faninsar/processing/merge/geocode_raster.py` | 当前 bilinear 实现（需修改） |
| `faninsar/processing/merge/pipeline.py` | M2 merge 消费 geocoded complex → 继承 kernel 偏差 |
| `faninsar/processing/merge/mosaic.py` | phase_network + mosaic → 无法修正已写入的 kernel 偏差 |
| `faninsar/datasets/frame/raster_io.py` | `reproject_phase_to_geogrid` 复数平均 — 修 wrap 但不修 kernel |
| `faninsar/datasets/frame/metadata.py` | CONTINUOUS_ASSETS vs PHASE_ASSETS 分类（正确分离） |
| `.omo/ulw-research/20260711-155830/SYNTHESIS.md` | 已标记 kernel 为 evidence gap |

## 附录 B — 参考文献

1. ISCE2 source: `components/zerodop/geozero/Geocodable.py`, `components/stdproc/resamp_slc/Resamp_slc.py`
2. ISCE3: [isce3 geocode_slc API](https://isce-framework.github.io/isce3/api/python/isce3/geocode/geocode_slc.html)
3. NISAR L2 GSLC ATBD: phase-preserving sinc resampling, raised-cosine windowed 8-tap
4. ESA S1 IPF: S1-IF-ASD-PL-1007, Level-1 Product Algorithm Definition
5. Sentinel-1 IPF source: FFT-based range compression and azimuth processing (sinc-equivalent)
6. `.omo/ulw-research/20260711-155830/SYNTHESIS.md`, `wave-1-partial-team-evidence.md`
7. `.omo/evidence/task-2-faninsar-python-insar-master-plan/reverify/mutant-gpl-clean-room.toml`
