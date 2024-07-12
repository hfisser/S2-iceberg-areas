import os
import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
from iceberg_detection.S2CFAR import to_polygons

THRESHOLD = 255 * 0.5

dir_dornier = "/media/henrik/DATA/raster/dornier_flight_campaign"

files_ocean = [
    os.path.join(dir_dornier, "ocean_s2/S2B_MSIL1C_20200621T121649_N0209_R009_T33XXH_20200621T141815_pB3_ocean.gpkg"),
    os.path.join(dir_dornier, "ocean_s2/S2B_MSIL1C_20200621T121649_N0209_R009_T33XWH_20200621T141815_pB3_ocean.gpkg")
]
aoi = gpd.read_file(os.path.join(dir_dornier, "aoi.gpkg"))


class DornierFlight:
    def __init__(self, dir_dornier) -> None:
        self.dir_dornier = dir_dornier
        self.dir_icebergs = os.path.join(self.dir_dornier, "dornier_reference_icebergs")
        self.meta = None
        self.file_icebergs = None

    def detect_icebergs(self):
        files_icebergs = []
        for line in range(1, 4):
            print("Line:", line)
            files_icebergs.append(self._detect_icebergs_line(line))
        icebergs = gpd.GeoDataFrame(pd.concat([gpd.read_file(file).geometry for file in files_icebergs]))
        icebergs.index = list(range(len(icebergs)))
        self.file_icebergs = os.path.join(self.dir_icebergs, "reference_icebergs_merged.gpkg")
        icebergs.to_file(self.file_icebergs)

    def _detect_icebergs_line(self, line):
        try:
            os.mkdir(self.dir_icebergs)
        except FileExistsError:
            pass
        data = self.read_data(line)
        outliers = np.int8(np.sum(data > THRESHOLD, 0) == 3)
        data = None
        print("Polygonizing")
        iceberg_polygons, _ = to_polygons(outliers, self.meta["transform"], self.meta["crs"])
        file_out = os.path.join(self.dir_icebergs, f"line-{line}.gpkg")
        iceberg_polygons.to_file(file_out)
        return file_out
        
    def read_data(self, line):
        with rio.open(os.path.join(self.dir_dornier, f"line-{line}.tif")) as src:
            data, self.meta = src.read(), src.meta
        return data


if __name__ == "__main__":
    dornier = DornierFlight(dir_dornier)
    dornier.detect_icebergs()
