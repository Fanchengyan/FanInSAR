from pathlib import Path

from faninsar import datasets, query

home_dir = Path("/Users/fancy/data")

ds_unw = datasets.HyP3S1(home_dir)
ds_coh = ds_unw.coh_dataset

# initialize a Points from a shape file, which contains reference points
ref_file = "/Volumes/Data/GeoData/YNG/ARPs.geojson"
ref_points = query.Points(
    [[-70.14524333, -35.92581708], [-70.41402755, -35.99540497]], crs="WGS84"
)

# define a bounding box for the region of interest
roi = query.BoundingBox(
    -70.69267819, -36.25849567, -70.21517087, -35.89980404, crs="WGS84"
)

# define a GeoQuery, which is a combination of a bounding box and a set of reference points
geo_query = query.GeoQuery(boxes=roi, points=ref_points)


pairs = ds_unw.pairs

mask_60 = pairs.days <= 60

pairs_used = pairs[mask_60]

# query the interferogram and coherence files with the selected pairs
unw_sample2 = ds_unw.query(geo_query, pairs=pairs_used)
coh_sample2 = ds_coh.query(geo_query, pairs=pairs_used)

unw = ds_unw.box_query(roi)

unw.data.sel(pair="20190419_20190501").plot()
