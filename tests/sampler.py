from pathlib import Path

from tqdm import tqdm

from faninsar.datasets import BoundingBox, RasterDataset
from faninsar.samplers.batch import RowSampler

home_dir = Path(r'E:\hyp3_result\ifgs')
files = list(home_dir.rglob('*unw_phase_clip.tif'))

ds = RasterDataset(file_paths=files[:5])


def test_row_sampler():
    for n in [10, 100, 1000]:
        sampler = RowSampler(ds, patch_num=n)
        print(len(list(sampler)))


def test_sample():
    points = [
        (490357, 4283413),
        (491048, 4283411),
        (490317, 4284829)
    ]
    points = ds.sample(points, verbose=True)
    print(points)


def test_bbox_index():
    bbox = BoundingBox(462954, 4258112, 543127, 4333928)
    sample = ds[bbox]
    print(sample)


def test_batch_sampler():
    sampler = RowSampler(ds, patch_num=10)
    samples = []
    for bbox in tqdm(sampler):
        samples.append(ds[bbox])
    return samples


# test_row_sampler()
# test_sample()
# test_bbox_index()
samples = test_batch_sampler()
