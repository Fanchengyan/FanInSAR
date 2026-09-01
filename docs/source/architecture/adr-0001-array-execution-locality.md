# ADR-0001: Array execution and locality contract

Status: accepted for the Python-native processing implementation.

## Decision

Every scientific operation has a NumPy or SciPy CPU reference implementation.1. IRLS
PyTorch is an optional kernel accelerator, not the owner of orchestration or
scientific truth. Dask builds and schedules task graphs around explicit
locality boundaries. xarray carries labelled dimensions and metadata; Zarr is
the restart-safe persistent boundary between major stages.

There is no universal `map_blocks` or `blockwise` adapter. Each operation
declares one of five dependency classes and uses its corresponding graph shape:

| Locality | Meaning | Dask graph form |
|---|---|---|
| elementwise | each output sample depends on the matching input sample | `map_blocks`, with shape/dtype declared |
| finite-halo | each output sample needs a bounded neighbourhood | `map_overlap`, explicit depth and boundary |
| tile-global | a complete tile or transform axis is one numerical domain | delayed tile task after validated rechunk |
| globally-coupled | all samples are linked through a solver or state | one bounded domain, decomposition, or staged iterative graph |
| reduction/network | associative reduction or graph/network solve | tree reduction or explicit graph-domain tasks |

An FFT transform axis must be one chunk before graph submission. A
globally-coupled operation is never silently converted to independent blocks.
Halo depth is included in peak-memory estimates. Rechunking is a named stage
with a durable output when its shuffle would otherwise dominate memory.

## Backend and dtype policy

The `scientific` policy uses float64/complex128 for geometry, orbit fitting,
phase models, reductions, solver state, and acceptance metrics. Stored science
rasters may use float32/complex64 only after a checked conversion. The
`accelerated` policy uses float32/complex64 for eligible CUDA kernels; reduction
accumulators remain float64 where reproducibility requires it. Backend choice,
input dtype, output dtype, deterministic mode, and tolerance are provenance.

CUDA workers advertise a Dask `GPU` resource and receive work only through
resource annotations. CPU workers do not pretend to satisfy that resource.
Transfers are batched at tile boundaries and buffers may be reused. Memory
release follows ownership/lifetime; `torch.cuda.empty_cache()` is not called
after every task.

MPS is a bounded, executable-probe capability, not CUDA parity. Initially only
complex multiplication, phase rotation, and window statistics may select MPS.
Unsupported operations fail preflight or take an explicitly selected CPU
fallback; they never fall back silently. MPS workers advertise `MPS`, not
`GPU`, because the memory and process models differ.

## Operation matrix

| Operation | Locality | Reference | Accelerator | Working dtype | Chunk rule | Explicit fallback |
|---|---|---|---|---|---|---|
| SAFE/EOPF metadata parse | reduction/network | Python/lxml | none | typed scalars | product/burst records | fail with source/remedy |
| SLC/calibration/noise read | elementwise | NumPy | none | complex64/float32 storage | source-aligned range tiles | local buffered read |
| orbit interpolation | tile-global | NumPy/SciPy | none | float64 | full required time stencil | CPU only |
| DEM sampling | finite-halo | NumPy/SciPy | CUDA after benchmark | float64 geometry | halo from interpolator | CPU reference |
| `rdr2geo` / `geo2rdr` solve | tile-global | NumPy/SciPy | CUDA after parity proof | float64 | independent bounded geo/radar tiles | CPU reference |
| transform LUT generation | tile-global | NumPy/SciPy | CUDA after parity proof | float64 | tile plus solver margin | CPU reference |
| calibration and noise correction | elementwise | NumPy | CUDA/MPS probe | float32/complex64 | source-aligned | CPU reference |
| TOPS deramp/reramp | elementwise | NumPy | CUDA/MPS phase rotation | complex64, float64 phase | burst-valid tiles | CPU reference |
| range/azimuth FFT filtering | tile-global | NumPy/SciPy | CUDA | complex64/128 | one chunk on transform axis | rechunk then CPU |
| coarse correlation | tile-global | NumPy/SciPy | CUDA after benchmark | float32/complex64 | complete search window | CPU reference |
| offset refinement | finite-halo | NumPy/SciPy | CUDA after benchmark | float64 fit | window plus search halo | CPU reference |
| burst resampling/coregistration | finite-halo | NumPy/SciPy | CUDA | complex64, float64 coordinates | interpolation halo, burst validity | CPU reference |
| interferogram formation | elementwise | NumPy | CUDA/MPS | complex64 | pair-aligned tiles | CPU reference |
| topographic phase removal | elementwise | NumPy | CUDA/MPS phase rotation | complex64, float64 phase | pair-aligned tiles | CPU reference |
| coherence/window statistics | finite-halo | NumPy/SciPy | CUDA/MPS probe | float32, float64 accumulation | window halo | CPU reference |
| phase filtering | finite-halo or tile-global FFT | NumPy/SciPy | CUDA | complex64 | filter halo or single FFT axis | CPU reference |
| connected components | reduction/network | NumPy/SciPy | none initially | int32 labels | tile labels then boundary merge | CPU only |
| IRLS phase unwrapping | globally-coupled | NumPy/SciPy sparse | CUDA operator after proof | float64 solver, float32 output | bounded component/domain decomposition | CPU reference |
| geocoded raster resampling | finite-halo | NumPy/SciPy | CUDA after proof | source dtype, float64 coordinates | target tile plus source halo | CPU reference |
| pair-network closure | reduction/network | NumPy/SciPy | none initially | float64 | explicit acquisition graph | CPU only |
| SBAS/NSBAS inversion | globally-coupled | NumPy/SciPy | Torch CUDA after parity | float64 reference | spatial tiles, full temporal axis | CPU reference |
| Zarr transaction/write | reduction/network | xarray/Zarr | none | declared asset dtype | storage-aligned chunks | local or object-store writer |
| STAC publication | reduction/network | Python/pystac | none | typed metadata | one item/collection task | fail before publish |

## Chunk and memory preflight

`plan_chunks` creates deterministic balanced chunks from array shape, element
size, and target input bytes. Algorithm-specific planners may replace it, but
must return an explicit `ChunkLayout`. Peak bytes include the largest expanded
input chunk plus every concurrently live output/temporary array. Worker usable
memory is `worker_bytes * safety_fraction`; spill headroom, interpreter/native
overhead, and concurrent task slots are excluded from that usable fraction.

Before submission, `validate_execution` rejects malformed layouts, multi-chunk
FFT axes, missing CUDA/MPS resources, operations outside the MPS subset, and a
tile whose estimated peak exceeds the worker budget. Production scheduling
profiles must additionally derive concurrent task count from the same budget;
Dask spilling is recovery headroom, not permission to submit an oversized
single task.

## Determinism and failure behavior

The CPU reference is the acceptance oracle. CUDA deterministic algorithms are
requested when supported and the selected mode is recorded. An operation that
cannot meet deterministic policy raises before execution instead of changing
the policy. Fixed chunk plans, stable reduction order, solver tolerances, seeds,
library versions, and device capability are captured in provenance.

Resource annotations are validated against scheduler worker metadata before
submission so an impossible `GPU` request cannot wait forever. Cancellation
occurs only at stage/tile boundaries. A completed Zarr stage is published by a
transaction marker after metadata validation; partial stores are not accepted
as cache hits on restart.

## Consequences

The CPU path remains slower but portable and testable. Accelerators can be
introduced operation by operation without changing product semantics. Global
algorithms require more deliberate graph construction than blockwise kernels,
but the contract prevents scientifically invalid chunk-local answers and
predictable scheduler hangs or worker OOM failures.
