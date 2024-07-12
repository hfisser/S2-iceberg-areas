import os
import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
from datetime import datetime
from rasterio.mask import mask
from shapely.geometry import box
from iceberg_detection.S2CFAR import to_polygons
from resources.parse_resources import get_constant
from iceberg_detection.S2IcebergDetector import S2IcebergDetector
from tools.copernicus_data.download_and_prepare_s2 import calc_sza

dir_dornier = "/media/henrik/DATA/raster/dornier_flight_campaign"
files_s2 = [
    "/media/henrik/DATA/raster/dornier_flight_campaign/S2/S2B_MSIL1C_20200621T121649_N0209_R009_T33XWH_20200621T141815_pB3.jp2",
    "/media/henrik/DATA/raster/dornier_flight_campaign/S2/S2B_MSIL1C_20200621T121649_N0209_R009_T33XXH_20200621T141815_pB3.jp2"
]
file_ocean = "/media/henrik/DATA/raster/dornier_flight_campaign/aoi.gpkg"
file_ws = "/media/henrik/DATA/raster/dornier_flight_campaign/CARRA/east_domain_surface_or_atmosphere_analysis_10m_wind_speed_2020_2020_6_6_21_21_12_12_33XWH_.tif"
dir_out = "/media/henrik/DATA/raster/dornier_flight_campaign/icebergs_s2"
date = datetime(2020, 6, 21, 12, 16, 49)
file_matches = os.path.join(dir_dornier, "matches/match_aois.gpkg")


def read_data(file_s2, file_ocean):
    detector = S2IcebergDetector()
    with rio.open(file_s2) as src:
        meta = src.meta
        data_s2, _ = mask(src, list(gpd.read_file(file_ocean).to_crs(src.crs).geometry), nodata=0, indexes=[1, 5])
        bounds = src.bounds
    data_s2 = data_s2.astype(np.float32, copy=False)
    data_s2[0, data_s2[-1] > 5] = np.nan
    data_s2 = data_s2[0]
    detector.meta = meta
    data_s2 *= 1e-4
    data_s2[data_s2 == 0] = np.nan
    aoi = gpd.GeoDataFrame(geometry=[box(bounds.left, bounds.bottom, bounds.right, bounds.top)], crs=meta["crs"])
    acq = dict(geometry=aoi.to_crs("EPSG:4326").geometry.iloc[0], startDate=date)
    return data_s2, acq, meta


def detect_icebergs(file_ws, acq, meta, threshold, dir_out):    
    detector = S2IcebergDetector()
    detector.meta = meta
    acq["startDate"] = date.strftime(get_constant("FORMAT_DATE_TIME"))
    outliers = detector.detect(data_s2, file_s2, file_ws, calc_sza(acq), threshold)
    iceberg_polygons, _ = to_polygons(outliers, detector.meta["transform"], detector.meta["crs"])
    file_out = os.path.join(dir_out, "iceberg_polygons_{0}_{1}.gpkg".format(os.path.basename(file_s2).split(".")[0], threshold))
    iceberg_polygons.to_file(file_out)
    return file_out, ""


if __name__ == "__main__":
    thresholds = np.round(np.arange(0.06, 0.23, 0.01), 2)
    files_out = {threshold: [] for threshold in thresholds}
    for file_s2 in files_s2:
        data_s2, acq, meta = read_data(file_s2, file_ocean)
        for threshold in thresholds:
            print("{0} // {1}".format(file_s2, threshold))
            file_out, _ = detect_icebergs(file_ws, acq, meta, threshold, dir_out)
            files_out[threshold].append(file_out)
    for threshold in thresholds:
        merged = gpd.GeoDataFrame(geometry=pd.concat([gpd.read_file(file_out).geometry for file_out in files_out[threshold]]), crs=meta["crs"])
        merged.to_file(os.path.join(dir_out, "merged", os.path.basename(files_out[threshold][0].replace(".gpkg", "_merged.gpkg"))))
        merged.index = list(range(len(merged)))
