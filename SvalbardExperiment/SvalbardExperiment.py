import os
import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
from plots.Plot import Plot
from decimal import Decimal
from rasterio.mask import mask
from shapely.geometry import box
from scipy.stats import linregress
from tools.vector_utils.vector_utils import close_holes
from tools.stats_utils.stats_utils import mean_confidence_interval
from tools.copernicus_data.download_and_prepare_s2 import calc_sza
from resources.parse_resources import get_color, get_parameter, get_constant, get_iceberg_area_classes

DPI = get_parameter("DPI_PLOTS_MEDIUM")

dir_plot = os.path.join(get_constant("DIR_PLOT"), "dornier_flight_comparison")
try:
    os.mkdir(dir_plot)
except FileExistsError:
    pass
dir_dornier = "/media/henrik/DATA/raster/dornier_flight_campaign"
dir_selected_s2 = os.path.join(dir_dornier, "icebergs_s2", "selected")
dir_s2 = os.path.join(dir_dornier, "S2")
file_matches = os.path.join(dir_dornier, "matches/match_aois.gpkg")
file_icebergs_dornier = os.path.join(dir_dornier, "dornier_reference_icebergs/icebergs_reference_selected_v1.gpkg")
file_icebergs_s2_selected_dummy = os.path.join(dir_dornier, "icebergs_s2/icebergs_s2_selected_v0.gpkg")
files_s2 = [
    "/media/henrik/DATA/raster/dornier_flight_campaign/S2/S2B_MSIL1C_20200621T121649_N0209_R009_T33XWH_20200621T141815_pB3.jp2",
    "/media/henrik/DATA/raster/dornier_flight_campaign/S2/S2B_MSIL1C_20200621T121649_N0209_R009_T33XXH_20200621T141815_pB3.jp2"
]
file_icebergs_s2_merged = "/media/henrik/DATA/raster/dornier_flight_campaign/icebergs_s2/merged/iceberg_polygons_S2B_MSIL1C_20200621T121649_N0209_R009_T33XWH_20200621T141815_pB3_0.20999999999999994_merged.gpkg"
files_icebergs_s2_merged = list(set(glob("{}*merged*gpkg".format(file_icebergs_s2_merged.split(".")[0]))) - set(glob("{0}{1}*merged*gpkg".format(file_icebergs_s2_merged.split(".")[0], "*refined"))))
#files_icebergs_s2_merged = glob("{}*refined*merged*gpkg".format(file_icebergs_s2_merged.split(".")[0]))


class S2DornierComparison:
    def __init__(self) -> None:
        self.icebergs_matched_dornier = None
        self.icebergs_matched_s2 = None
        self.areas = pd.DataFrame()
    
    def filter_s2_icebergs(self, file_icebergs_s2_merged, file_icebergs_s2_selected, files_s2, dir_out):
        print("Filtering:", file_icebergs_s2_merged)
        icebergs_s2_selected = gpd.read_file(file_icebergs_s2_selected)
        icebergs_s2 = gpd.read_file(file_icebergs_s2_merged)
        icebergs_s2.geometry = icebergs_s2.geometry.apply(lambda p: close_holes(p)).make_valid()
        icebergs_s2.to_file(file_icebergs_s2_merged)
        icebergs_s2.geometry = icebergs_s2.make_valid()
        icebergs_s2_filtered = gpd.GeoDataFrame()
        for i, iceberg_selected in icebergs_s2_selected.iterrows():
            iceberg_s2 = icebergs_s2.clip(iceberg_selected["geometry"].centroid.buffer(500))
            if len(iceberg_s2) > 1:
                for j, ib in iceberg_s2.iterrows():
                    iceberg_s2.loc[j, "dist"] = ib["geometry"].centroid.distance(iceberg_selected["geometry"].centroid)
                iceberg_s2 = iceberg_s2.loc[iceberg_s2["dist"].idxmin()]
            elif len(iceberg_s2) == 1:
                iceberg_s2 = iceberg_s2.iloc[0]
            else:
                pass
            if len(iceberg_s2) > 0:
                idx = len(icebergs_s2_filtered)
                icebergs_s2_filtered.loc[idx, "geometry"] = iceberg_s2["geometry"]
                try:
                    data = self.read_in_aoi(files_s2[0], iceberg_s2)
                except ValueError:
                    data = self.read_in_aoi(files_s2[1], iceberg_s2)
                icebergs_s2_filtered.loc[idx, "B8_mean"] = np.nanmean(data)
                icebergs_s2_filtered.loc[idx, "B8_p25"] = np.nanpercentile(data, 25)
                icebergs_s2_filtered.loc[idx, "B8_p75"] = np.nanpercentile(data, 75)
                icebergs_s2_filtered.loc[idx, "B8_min"] = np.nanmin(data)
                icebergs_s2_filtered.loc[idx, "B8_max"] = np.nanmax(data)
                icebergs_s2_filtered.loc[idx, "B8_std"] = np.nanstd(data)
        icebergs_s2_filtered.index = list(range(len(icebergs_s2_filtered)))
        icebergs_s2_filtered.geometry = icebergs_s2_filtered.geometry
        icebergs_s2_filtered.crs = icebergs_s2.crs
        file_filtered = os.path.join(dir_out, os.path.basename(file_icebergs_s2_merged).replace(".gpkg", "_selected_v0.gpkg"))
        icebergs_s2_filtered.to_file(file_filtered)
        return file_filtered

    def match(self, file_matches, file_icebergs_dornier, file_icebergs_s2):
        print("Reading")
        icebergs_dornier = gpd.read_file(file_icebergs_dornier)
        icebergs_dornier.geometry = icebergs_dornier.geometry.apply(lambda p: close_holes(p))
        icebergs_dornier.to_file(file_icebergs_dornier)
        icebergs_s2 = gpd.read_file(file_icebergs_s2)
        match_aois = gpd.read_file(file_matches)
        print("Identifying matches")
        self.icebergs_matched_dornier = self.identify_matches(icebergs_dornier, match_aois)
        self.icebergs_matched_s2 = self.identify_matches(icebergs_s2, match_aois)
        self.extract_areas(np.round(float(file_icebergs_s2.replace("_refined", "").split("_")[-4]), 2))

    def extract_reflectance_s2(self, files_s2, file_icebergs):
        nir = []
        with rio.open(files_s2[0]) as src0:
            with rio.open(files_s2[1]) as src1:
                for i, iceberg in gpd.read_file(file_icebergs):
                    try:
                        data, _ = mask(src0, [iceberg["geometry"]], crop=True, nodata=np.nan, indexes=1)
                    except ValueError:
                        data, _ = mask(src1, [iceberg["geometry"]], crop=True, nodata=np.nan, indexes=1)
                    nir.append(data.flatten())

    def extract_areas(self, threshold):
        k = 0
        for i, iceberg_dornier in self.icebergs_matched_dornier.iterrows():
            for j, iceberg_s2 in self.icebergs_matched_s2.iterrows():
                if iceberg_dornier["index_match"] == iceberg_s2["index_match"]:
                    area_dornier = iceberg_dornier["geometry"].area
                    area_s2 = iceberg_s2["geometry"].area
                    re = self.calc_relative_error(np.float32(area_s2), np.float32(area_dornier))
                    self.icebergs_matched_s2.loc[j, "relative_error"] = re
                    if any([re < -60, re > 45]) and threshold == 0.12:
                        print(re)
                        print(np.sqrt(area_s2))
                        print(np.sqrt(area_dornier))
                        iceberg_s2["re"] = re
                    self.areas.loc[k, f"area_dornier_threshold_{threshold}"] = area_dornier
                    self.areas.loc[k, f"area_s2_threshold_{threshold}"] = area_s2
                    self.areas.loc[k, f"index_match_threshold_{threshold}"] = iceberg_dornier["index_match"]
                    for column in iceberg_s2.keys():
                        if "B8" in column:
                            self.areas.loc[k, f"{column}_threshold_{threshold}"] = iceberg_s2[column]
                    k += 1
        if threshold == 0.12:
            self.icebergs_matched_s2.to_file(os.path.join(dir_selected_s2, "icebergs_s2_relative_error_dornier.gpkg"))

    def compare(self, dir_plot):
        print("Comparing")
        column_dornier = "area_dornier_threshold_"
        column_s2 = column_dornier.replace("dornier", "s2")
        thresholds = np.sort(np.unique([float(column.split("threshold_")[-1]) for column in self.areas.columns]))
        area_stats = pd.DataFrame()
        relative_errors_median = []
        for i, threshold in enumerate(thresholds):
            areas_dornier, areas_s2 = self.areas[f"{column_dornier}{threshold}"], self.areas[f"{column_s2}{threshold}"]
            not_nan = ~np.isnan(areas_dornier)
            areas_dornier, areas_s2 = areas_dornier[not_nan], areas_s2[not_nan]
            relative_error = self.calc_relative_error(areas_s2, areas_dornier)
            absolute_error = areas_s2 - areas_dornier
            p25, p75 = np.percentile(relative_error, 25), np.percentile(relative_error, 75)
            p10, p90 = np.percentile(relative_error, 10), np.percentile(relative_error, 90)
            relative_errors_median.append(np.median(relative_error))
            area_stats.loc[i, "median_relative_error"] = relative_errors_median[-1]
            area_stats.loc[i, "mean_relative_error"] = np.mean(relative_error)
            area_stats.loc[i, "mean_absolute_error"] = np.mean(absolute_error)
            ci = mean_confidence_interval(relative_error, 0.95)[1]
            area_stats.loc[i, "ci95_low"] = ci[0]
            area_stats.loc[i, "ci95_high"] = ci[1]
            area_stats.loc[i, "p25_relative_error"] = p25
            area_stats.loc[i, "p75_relative_error"] = p75
            area_stats.loc[i, "p10_relative_error"] = p10
            area_stats.loc[i, "p90_relative_error"] = p90
            area_stats.loc[i, "std_relative_error"] = np.std(relative_error)
            area_stats.loc[i, "max_relative_error"] = np.max(relative_error)
            area_stats.loc[i, "min_relative_error"] = np.min(relative_error)
            area_stats.loc[i, "proportion_detected"] = len(areas_s2) / len(self.areas) * 100            
            area_stats.loc[i, "mae"] = np.mean(np.abs(np.subtract(areas_s2, areas_dornier)))
            area_stats.loc[i, "threshold"] = threshold
        self.plot_reflectance_by_error()
        self.plot_size_distributions()
        print("_" * 100)
        print("Sensitivity")
        self.plot_sensitivity(area_stats, dir_plot)
        print("_" * 100)
        print("Best threshold scatter")
        self.plot_selected_threshold_scatter(area_stats, dir_plot)
        print("_" * 100)
        print("RE by size")
        self.plot_relative_error_by_size(area_stats, dir_plot)
        print("_" * 100)
        return thresholds[np.argmin(np.abs(relative_errors_median))]

    def plot_reflectance_by_error(self):
        threshold = "0.12"
        column_dornier = "area_dornier_threshold_"
        column_s2 = column_dornier.replace("dornier", "s2")
        b8_mean = np.float32(self.areas[f"B8_mean_threshold_{threshold}"])
        area_dornier, area_s2 = self.areas[f"{column_dornier}{threshold}"], self.areas[f"{column_s2}{threshold}"]
        print("Max RL Dornier: ", np.max(np.sqrt(area_dornier)))
        not_nan = ~np.isnan(area_dornier)
        area_dornier, area_s2 = area_dornier[not_nan], area_s2[not_nan]
        b8_mean = b8_mean[not_nan]
        relative_error = self.calc_relative_error(area_s2, area_dornier)
        bins = np.float32([0, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35])
        binned = self.group(relative_error, b8_mean, bins)
        stats = dict(p25=[], p50=[], p75=[], mean=[])
        for key, values in binned.items():
            stats["p25"].append(np.nanpercentile(values, 25))
            stats["p50"].append(np.nanpercentile(values, 50))
            stats["p75"].append(np.nanpercentile(values, 75))
            stats["mean"].append(np.nanmean(values))
        plot = Plot(dir_plot, DPI, get_color("iceberg"))
        plot.create_figure(1, 1, (4.5, 3))
        plot.axes.plot(bins, stats["mean"], color=get_color("iceberg"), linewidth=2, label="Mean RE")
        plot.axes.fill_between(bins, stats["p25"], stats["p75"], color=plot.color, alpha=0.3)
        plot.axes.plot([plot.axes.get_xlim()[0], plot.axes.get_xlim()[1]], [0, 0], linestyle="dotted", color="black")
        n = np.int16([len(x) for x in binned.values()])
        labels, handles = plot.axes.get_legend_handles_labels()
        plot.axes.set_ylabel("RE (%)")
        plot.axes.set_xlabel(r"$\rho_{NIR}$")
        plot.fig.legend(np.hstack([labels]), np.hstack([handles]), loc="lower center", ncol=3, fontsize=9)
        plot.minimalistic_layout(plot.axes)
        plot.fig.subplots_adjust(bottom=0.26)
        plot.save_figure(f"b8_mean_by_relative_error_threshold_{threshold}")
    
    def plot_size_distributions(self):
        areas_dornier = np.sqrt(self.areas["area_dornier_threshold_0.12"])
        areas_s2 = np.sqrt(self.areas["area_s2_threshold_0.12"])
        plot = Plot(dir_plot, DPI, get_color("iceberg"))
        plot.create_figure(1, 2, (6.5, 3))
        plot.axes[0].hist(areas_dornier**2, bins=10, color="#4ace51")
        plot.axes[1].hist(areas_s2**2, bins=10, color=get_color("iceberg"))
        plot.axes[0].set_ylabel("Count")
        plot.axes[0].set_title("a")
        plot.axes[1].set_title("b")
        plot.axes[1].set_yticklabels([])
        for i, ax in enumerate(plot.axes):
            xlabel = [r"$A_{DO}$ (m²)", r"[$\sqrt{A_{DO}}$ (m)]"] if i == 0 else [r"$A_{S2}$ (m²)", r"[$\sqrt{A_{S2}}$ (m)]"]
            ax.set_xlabel("\n".join(xlabel))
            ticks = [0, 60**2, 122**2, 160**2]
            ax.set_xlim(0, np.max(ticks))
            ax.set_xticks(ticks)
            ticklabels = ["%.1E" % Decimal(float(tick)) for tick in ticks]
            ticklabels = ["{0}\n[{1}]".format(ticklabel, int(np.round(np.sqrt(tick)))) for ticklabel, tick in zip(ticklabels, ticks)]
            ticklabels[0] = "0"
            ax.set_xticklabels(ticklabels)
            plot.minimalistic_layout(ax)
        plot.save_figure("dornier_sentinel2_size_distribution")
        for platform, values in zip(["Dornier", "S2"], [areas_dornier, areas_s2]):
            mean_ci = mean_confidence_interval(values, 0.95)
            print("{0} ({1}) & {2} & {3} & {4} & {5} & {6}".format(
                np.round(mean_ci[0], 2),
                np.round(np.std(values), 2),
                np.round(np.nanpercentile(values, 25), 2),
                np.round(np.nanpercentile(values, 50), 2),
                np.round(np.nanpercentile(values, 75), 2),
                np.round(np.nanmin(values), 2),
                np.round(np.nanmax(values), 2)
            ))
        not_detected = np.sqrt(self.areas.loc[67:, "area_dornier_threshold_0.06"])
        print("Not detected", not_detected)
        print("Not detected mean sqrt area", np.mean(not_detected))
        print("Not detected max sqrt area", np.max(not_detected))
        print("Not detected min sqrt area", np.min(not_detected))

    def plot_selected_threshold_scatter(self, area_stats, dir_plot):
        column_dornier = "area_dornier_threshold_"
        column_s2 = column_dornier.replace("dornier", "s2")
        threshold = area_stats.loc[area_stats["mean_relative_error"].abs().idxmin(), "threshold"]
        print("Best threshold:", threshold)
        areas_dornier, areas_s2 = self.areas[f"{column_dornier}{threshold}"], self.areas[f"{column_s2}{threshold}"]
        not_nan = ~np.isnan(areas_dornier)
        areas_dornier, areas_s2 = areas_dornier[not_nan], areas_s2[not_nan]
        print("Max RL Dornier:", np.max(np.sqrt(areas_dornier)))
        relative_error = self.calc_relative_error(areas_s2, areas_dornier)
        plot = Plot(dir_plot, get_parameter("DPI_PLOTS_MEDIUM"), get_color("iceberg"))
        plot.create_figure(1, 1, (5.75, 5))
        #norm = Normalize(vmin=-5, vmax=5)
        plot.axes.scatter(areas_s2, areas_dornier, s=35, alpha=0.6, color=get_color("iceberg"), edgecolors="none")
        plot.axes.set_xlabel("\n".join([r"$A_{S2}$ (m²)", r"[$\sqrt{A_{S2}}$ (m)]"]))
        plot.axes.set_ylabel("\n".join([r"$A_{DO}$ (m²)", r"[$\sqrt{A_{DO}}$ (m)]"]), rotation=0, labelpad=35)
        ticks = [0, 60**2, 122**2, 160**2]
        plot.axes.set_xlim(0, np.max(ticks))
        plot.axes.set_ylim(0, np.max(ticks))
        plot.axes.set_xticks(ticks)
        plot.axes.set_yticks(ticks)
        ticklabels = ["%.1E" % Decimal(float(tick)) for tick in ticks]
        ticklabels = ["{0}\n[{1}]".format(ticklabel, int(np.round(np.sqrt(tick)))) for ticklabel, tick in zip(ticklabels, ticks)]
        ticklabels[0] = "0"
        plot.axes.set_xticklabels(ticklabels)
        plot.axes.set_yticklabels(ticklabels)
        regression = linregress(areas_s2, areas_dornier)
        regression_txt = "y = {0} x {1} {2}".format(np.round(regression.slope, 2), "-" if regression.intercept < 0 else "+", np.abs(np.round(regression.intercept, 2)))
        rvalue_txt = "Pearson's r-value: {}".format(np.round(regression.rvalue, 3))
        plot.axes.text(3000, 23000, "\n".join([regression_txt, rvalue_txt]), verticalalignment="top", bbox=dict(facecolor="white", alpha=0.5), fontsize=10)
        for location in ["right", "top"]:
            plot.axes.spines[location].set_visible(False)
        plot.fig.tight_layout()
        plot.save_figure("s2_vs_dornier_best_threshold_scatter")

    def plot_relative_error_by_size(self, area_stats, dir_plot):
        column_dornier = "area_dornier_threshold_"
        column_s2 = column_dornier.replace("dornier", "s2")
        threshold = area_stats.loc[area_stats["mean_relative_error"].abs().idxmin(), "threshold"]
        areas_dornier, areas_s2 = self.areas[f"{column_dornier}{threshold}"], self.areas[f"{column_s2}{threshold}"]
        not_nan = ~np.isnan(areas_dornier)
        areas_dornier, areas_s2 = areas_dornier[not_nan], areas_s2[not_nan]
        relative_error = self.calc_relative_error(areas_s2, areas_dornier)
        ae = np.abs(areas_s2 - areas_dornier)
        print("Mean relative error:", mean_confidence_interval(relative_error))        
        print("Mean absolute error:", np.mean(ae))
        print("Std absolute error", np.std(ae))
        print(np.percentile(ae, 25))
        print(np.percentile(ae, 75))
        print("Q1 AE", np.percentile(ae, 25))
        print("Q3 AE", np.percentile(ae, 75))
        print("Max positive relative error:", np.max(relative_error))
        print("Max negative relative error:", np.min(relative_error))
        print("P25 RE:", np.percentile(relative_error, 25))
        print("P75 RE:", np.percentile(relative_error, 75))
        print("Inner-quartile range:", np.percentile(relative_error, 75) - np.percentile(relative_error, 25))
        area_bins = np.hstack([0, get_iceberg_area_classes()])
        errors_grouped = self.group(relative_error, areas_dornier, area_bins)
        keys_delete = []
        for key, values in errors_grouped.items():
            if len(values) == 0:
                keys_delete.append(key)
        for key in keys_delete:
            del errors_grouped[key]
        errors_grouped_mean = [np.mean(values) for values in errors_grouped.values()]
        errors_grouped_p25 = [np.percentile(values, 25) for values in errors_grouped.values()]
        errors_grouped_p75 = [np.percentile(values, 75) for values in errors_grouped.values()]
        errors_grouped_p10 = [np.percentile(values, 10) for values in errors_grouped.values()]
        errors_grouped_p90 = [np.percentile(values, 90) for values in errors_grouped.values()]
        print("Mean RE grouped by size", errors_grouped_mean)
        print("Std RE by size", [np.std(values) for values in errors_grouped.values()])
        print("Q1-Q3 RE range by size", np.subtract(errors_grouped_p75, errors_grouped_p25))
        print("P10-P90 RE range by size", np.subtract(errors_grouped_p90, errors_grouped_p10))
        plot = Plot(dir_plot, get_parameter("DPI_PLOTS_MEDIUM"), color=get_color("iceberg"))
        plot.create_figure(1, 1, (6, 3))
        xvalues = list(errors_grouped.keys())
        plot.axes.set_xlabel("\n".join([r"$A_{DO}$ (m²)", r"$\sqrt{A_{DO}}$ (m)"]))
        plot.axes.set_ylabel("RE (%)")
        plot.axes.plot(xvalues, np.zeros(len(xvalues)), linestyle="dotted", color="black")
        plot.axes.plot(xvalues, errors_grouped_mean, linewidth=3.5, color=get_color("iceberg"), label="RE mean")
        plot.axes.set_ylim(-30, 30)
        xticklabels = ["{0}\n{1}".format(int(xvalue), int(np.round(np.sqrt(xvalue)))) for xvalue in xvalues]
        plot.axes.set_xticks(xvalues)
        plot.axes.set_xticklabels(xticklabels)
        #plot.axes.scatter(xvalues, relative_error, s=5, color=plot.color)
        for key, values in errors_grouped.items():
            print(key, "Mean", mean_confidence_interval(values, 0.9))
            print(key, "Median", np.median(values))
            print(key, "Inner-quartile range", np.percentile(values, 75) - np.percentile(values, 25))
        plot.axes.fill_between(list(errors_grouped.keys()), errors_grouped_p10, errors_grouped_p90, color=get_color("iceberg"), alpha=0.3)
        plot.axes.fill_between(list(errors_grouped.keys()), errors_grouped_p25, errors_grouped_p75, color=get_color("iceberg"), alpha=0.3)
        n = np.int16([len(x) for x in errors_grouped.values()])
        print("N samples size categories:", n)
        labels, handles = plot.axes.get_legend_handles_labels()
        #plot.fig.legend(np.hstack([labels]), np.hstack([handles]), loc="top right", ncol=3, fontsize=9)
        for location in ["left", "right", "top"]:
            plot.axes.spines[location].set_visible(False)
        plot.fig.tight_layout()
        plot.fig.subplots_adjust(bottom=0.3) 
        plot.save_figure("relative_error_by_size")
        self.plot_error_range_by_edge_proportion(xvalues, errors_grouped)

    def plot_error_range_by_edge_proportion(self, xvalues, errors_grouped):
        sizes = [int(np.mean([x, xvalues[i + 1]]) if i < (len(xvalues) - 1) else x) for i, x in enumerate(xvalues)]
        edge_proportion = np.zeros(len(sizes))
        for i, size in enumerate(sizes):
            a = np.arange(size**2).reshape((size, size))
            a[1:-1, 1:-1] = 0
            edge_proportion[i] = np.count_nonzero(a) / np.size(a) * 100
        plot = Plot(dir_plot, get_parameter("DPI_PLOTS_MEDIUM"), get_color("iceberg"))
        plot.create_figure(1, 2, (6, 3))
        argsorted = np.argsort(sizes)
        twinx = plot.axes[1]
        twinx.set_ylabel("RE range")
        p10_p90_range = np.subtract([np.percentile(values, 90) for values in errors_grouped.values()], [np.percentile(values, 10) for values in errors_grouped.values()])
        twinx.plot(xvalues, p10_p90_range, color=get_color("iceberg"), linewidth=3.5, label="P10-P90 range")
        twinx.set_ylim(0, 50)
        plot.axes[0].plot(xvalues, edge_proportion[argsorted], linewidth=3.5, color=get_color("iceberg"), label="Edge proportion")
        plot.axes[0].set_ylabel("Edge proportion of A (%)")
        plot.axes[0].set_ylim(0, 15)
        plot.axes[0].set_title("Edge proportion")
        plot.axes[1].set_title("Area error P10-P90 range")
        for ax in plot.axes:
            ax.set_xlabel(r"$\sqrt{A_{DO}}$ (m)")
            ax.set_xticks(xvalues)
            ax.set_xticklabels([int(x) for x in xvalues])
            plot.axes[0].set_yticklabels(np.int16(plot.axes[0].get_yticks()))
            plot.minimalistic_layout(ax)
        plot.save_figure("error_range_by_edge_proportion")
    
    def plot_selected_threshold_histogram(self, files_icebergs_s2, files_ocean, files_s2, dir_plot, threshold):
        data_s2 = {key: [] for key in ["icebergs", "edge", "ocean"]}
        for file_icebergs in files_icebergs_s2:
            icebergs = gpd.read_file(file_icebergs)
            for file_s2 in files_s2:
                try:
                    data_s2["icebergs"].append(self.read_in_aoi(file_s2, icebergs))
                except ValueError:
                    continue
        data_s2["icebergs"] = np.hstack(data_s2["icebergs"])
        xlabel = r"$\rho_{NIR}$"
        plot = Plot(dir_plot, get_parameter("DPI_PLOTS_MEDIUM"), get_color("iceberg"))
        plot.plot_histogram(
            data_s2["icebergs"],
            xlabel,
            "",
            40,
            percentage=True,
            alpha=1
        )
        plot.axes.set_xlim(0, 0.6)
        print("Mean", np.nanmean(data_s2["icebergs"]))
        print("Std", np.nanstd(data_s2["icebergs"]))
        print("P25", np.nanpercentile(data_s2["icebergs"], 25))
        print("P75", np.nanpercentile(data_s2["icebergs"], 75))
        if 0:
            plot.color = get_color("edge")
            plot.plot_histogram(
                data_s2["edge"],
                xlabel,
                "",
                40,
                percentage=True,
                alpha=0.6
            )
            plot.color = get_color("background")
            plot.plot_histogram(
                data_s2["ocean"],
                xlabel,
                "SZA~55°",
                40,
                percentage=True,
                alpha=0.6
            )
            plot.axes.legend(["Icebergs", "Iceberg edge", "Ocean"])
        plot.axes.set_xlim(0, 1)
        ymax = plot.axes.get_ylim()[1]
        plot.fig.set_size_inches((3.5, 3))
        plot.minimalistic_layout(plot.axes)
        plot.save_figure(f"s2_iceberg_histograms_threshold_{threshold}")


    @staticmethod
    def read_in_aoi(file_s2, aoi):
        with rio.open(file_s2) as src:
            try:
                data_s2, _ = mask(src, list(aoi["geometry"]), nodata=0, crop=True, indexes=1)
            except TypeError:
                data_s2, _ = mask(src, [aoi["geometry"]], nodata=0, crop=True, indexes=1)
            data_s2 = data_s2.astype(np.float32, copy=False)
            #data_s2[0, data_s2[-1] > 5] = np.nan
            data_s2[data_s2 == 0] = np.nan
            data_s2 *= 1e-4
        return data_s2.flatten()

    @staticmethod
    def group(x, group_reference, bins):
        grouped = dict.fromkeys(bins)
        for i, bin_value in enumerate(bins):
            try:
                next_value = bins[i + 1]
            except IndexError:
                next_value = np.inf
            grouped[bin_value] = x[np.bool8(group_reference >= bin_value) * np.bool8(group_reference < next_value)]
        return grouped

    @staticmethod
    def plot_sensitivity(area_stats, dir_plot):
        idx = area_stats["mean_relative_error"].abs().argmin()
        t = area_stats["threshold"].iloc[idx]        
        thresholds = np.float32(area_stats["threshold"])
        plot = Plot(dir_plot, get_parameter("DPI_PLOTS_MEDIUM"), get_color("iceberg"))
        plot.create_figure(1, 1, (6, 3))
        plot.axes.plot(area_stats["threshold"], area_stats["mean_relative_error"], linewidth=3.5, color=get_color("iceberg"), label="RE mean")
        plot.axes.plot(thresholds, np.zeros(len(thresholds)), linestyle="dotted", color="black")
        plot.axes.set_xlabel(r"Sentinel-2 $\rho_{NIR}$ threshold (reflectance)")
        plot.axes.set_ylabel("RE (%)")
        plot.axes.set_ylim(-70, 70)
        plot.axes.set_yticks(np.arange(plot.axes.get_ylim()[0], plot.axes.get_ylim()[1], 10))
        m = np.float32(area_stats["mean_relative_error"])
        iq_range = area_stats["p75_relative_error"] - area_stats["p25_relative_error"]
        p90_p10_range = area_stats["p90_relative_error"] - area_stats["p10_relative_error"]
        print("Mean REs", m)
        print("Inner-quartile range", iq_range)
        print("Mean inner-quartile range", np.mean(iq_range))
        print("Std inner-quartile range", np.std(iq_range))
        print("Minimum inner-quartile range", np.min(iq_range), np.argmin(iq_range) == idx)
        print("Maximum inner-quartile range", np.max(iq_range), thresholds[np.argmax(iq_range)])
        print("P10-P90 range", p90_p10_range)
        print("Mean P10-P90 range", np.mean(p90_p10_range))
        print("Std P10-P90 range", np.std(p90_p10_range))
        print("Minimum P10-P90 range", np.min(p90_p10_range), thresholds[np.argmin(p90_p10_range)])
        print("Maximum P10-P90 range", np.max(p90_p10_range), thresholds[np.argmax(p90_p10_range)])
        print(thresholds[np.argsort(p90_p10_range)])
        print("Mean RE at best threshold", mean_confidence_interval(np.float32(area_stats["mean_relative_error"])))
        print(area_stats["proportion_detected"][area_stats["threshold"] == t])
        print(np.subtract(area_stats["p75_relative_error"][area_stats["threshold"] == t], area_stats["p25_relative_error"][area_stats["threshold"] == t]))
        print(np.min(np.abs(np.subtract(area_stats["p75_relative_error"], area_stats["p25_relative_error"]))))
        print(area_stats["min_relative_error"][area_stats["threshold"] == t])
        print(area_stats["max_relative_error"][area_stats["threshold"] == t])
        print(area_stats["mae"][area_stats["threshold"] == t])
        diff = np.hstack([m[1:], np.nan]) - m
        mean_difference_per_step = np.nanmean(diff)
        print("Mean difference per threshold step", mean_difference_per_step)
        print("P25 difference per step", np.nanpercentile(diff, 25))
        print("P75 difference per step", np.nanpercentile(diff, 75))
        print("Std difference per step", np.nanstd(diff))
        plot.axes.fill_between(thresholds, np.float32(area_stats["p10_relative_error"]), np.float32(area_stats["p90_relative_error"]), alpha=0.3, color=get_color("iceberg"))
        plot.axes.fill_between(thresholds, np.float32(area_stats["p25_relative_error"]), np.float32(area_stats["p75_relative_error"]), alpha=0.3, color=get_color("iceberg"))
        best = area_stats.loc[idx]
        plot.axes.scatter(best["threshold"], best["mean_relative_error"], s=50, color=get_color("s2"), alpha=1, zorder=5)
        plot.axes.set_xticks(thresholds[::2])
        plot.minimalistic_layout(plot.axes)
        plot.fig.legend(loc="upper right", ncol=1, fontsize=9, bbox_to_anchor=(0.95, 0.9))
        plot.save_figure("s2_vs_dornier_relative_error_sensitivity")
        plt.close(plot.fig)

    def identify_matches(self, icebergs, match_aois):
        icebergs_matched = gpd.GeoDataFrame()
        for idx_match, match in match_aois.iterrows():
            for i, iceberg in icebergs.iterrows():
                if iceberg["geometry"].intersects(match["geometry"]):
                    icebergs_matched.loc[idx_match, "geometry"] = iceberg["geometry"]
                    icebergs_matched.loc[idx_match, "index_match"] = idx_match
                    icebergs_matched.loc[idx_match, "index_source"] = i
                    for col in iceberg.keys():
                        if "B8" in col:
                            icebergs_matched.loc[idx_match, col] = iceberg[col]
        icebergs_matched.geometry = icebergs_matched.geometry
        icebergs_matched.crs = icebergs.crs
        return icebergs_matched

    @staticmethod
    def calc_relative_error(areas_s2, areas_dornier):
        return np.float32(areas_s2 / areas_dornier * 100 - 100)


if __name__ == "__main__":
    for file in files_s2:
        with rio.open(file) as src:
            bounds = src.bounds
            geom = gpd.GeoDataFrame(geometry=[box(bounds.left, bounds.bottom, bounds.right, bounds.top)], crs=src.crs).to_crs("EPSG:4326").centroid.iloc[0]
        date = os.path.basename(file).split("_")[2]
        acq = dict(geometry=geom, startDate="T".join(["-".join([date[:4], date[4:6], date[6:8]]), ":".join([date[9:11], date[11:13], date[13:15]])]))
        print("SZA", calc_sza(acq))

    comparison = S2DornierComparison()
    files_icebergs_s2_selected = [comparison.filter_s2_icebergs(file_icebergs_s2_merged, file_icebergs_s2_selected_dummy, files_s2, dir_selected_s2) for file_icebergs_s2_merged in files_icebergs_s2_merged]
    for file in files_icebergs_s2_selected:
        comparison.match(file_matches, file_icebergs_dornier, file)
    best_threshold = comparison.compare(dir_plot)

    files_icebergs = [
        os.path.join(dir_dornier, f"icebergs_s2/iceberg_polygons_S2B_MSIL1C_20200621T121649_N0209_R009_T33XXH_20200621T141815_pB3_{best_threshold}.gpkg"),
        os.path.join(dir_dornier, f"icebergs_s2/iceberg_polygons_S2B_MSIL1C_20200621T121649_N0209_R009_T33XWH_20200621T141815_pB3_{best_threshold}.gpkg")
    ]
    files_ocean = [
        os.path.join(dir_dornier, "ocean_s2/S2B_MSIL1C_20200621T121649_N0209_R009_T33XXH_20200621T141815_pB3_ocean.gpkg"),
        os.path.join(dir_dornier, "ocean_s2/S2B_MSIL1C_20200621T121649_N0209_R009_T33XWH_20200621T141815_pB3_ocean.gpkg")
    ]

    files_icebergs_filtered = [comparison.filter_s2_icebergs(file_icebergs, file_icebergs_s2_selected_dummy, files_s2, dir_selected_s2) for file_icebergs in files_icebergs]
    files_s2 = [glob(os.path.join(dir_s2, "*{}*.jp2".format(os.path.basename(file).split("_")[-6])))[0] for file in files_icebergs_filtered]
