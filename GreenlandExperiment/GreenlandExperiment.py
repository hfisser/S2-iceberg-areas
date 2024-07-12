import os
import numpy as np
import rasterio as rio
import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
from plots.Plot import Plot
from datetime import datetime
from rasterio.mask import mask
from shapely.geometry import box
from scipy.stats import skew
from scipy.interpolate import interp1d
from iceberg_detection.S2CFAR import to_polygons
from tools.vector_utils.vector_utils import close_holes
from tools.stats_utils.stats_utils import mean_confidence_interval
from tools.copernicus_data.download_and_prepare_s2 import calc_sza
from resources.parse_resources import get_parameter, get_constant, get_color

THETA = r"$\theta$ (°)"
MEAN_ERROR_FROM_CALIBRATION = 0.19
P25_ERROR_FROM_CALIBRATION = -4.040286064147949
P75_ERROR_FROM_CALIBRATION = 6.858802318572998
BEST_THRESHOLD_FROM_CALIBRATION = 0.12

dir_s2 = "/media/henrik/DATA/raster/s2/25WER"
dir_plot = os.path.join(get_constant("DIR_PLOT"), "s2_sza_analysis")
dir_main = "/home/henrik/Output/icebergs/s2_sza_analysis"
dir_aois = os.path.join(dir_main, "aois")
dir_carra = get_constant("DIR_CARRA")


class S2SZA:
    def __init__(self, thetas, dir_main, dir_aois, dir_carra) -> None:
        self.buffer_distance = 100
        self.thetas = thetas
        self.theta_labels = [f"{theta}" for theta in thetas]
        self.dir_main = dir_main
        self.dir_aois = dir_aois
        self.dir_carra = dir_carra
        self.files_out = dict()
        self.data_icebergs = None
        self.data_neighborhoods = None
        self.data_ocean = None
        self.stats_at = dict(mean=[], ci_low=[], ci_high=[], p10=[], p25=[], p50=[], p75=[], p90=[], std=[], min=[], max=[])  # air temperature

    def read_data(self, files_icebergs, files_s2):        
        self.data_icebergs = dict.fromkeys(files_icebergs)
        self.data_all = {file: [] for file in files_icebergs}
        self.data_neighborhoods = dict.fromkeys(files_icebergs)
        self.data_ocean = dict.fromkeys(files_icebergs)
        sizes = []
        sizes_all = []
        for file_icebergs, file_s2 in zip(files_icebergs, files_s2):
            for buffer, key, file_aoi in zip([0, self.buffer_distance, 0], ["iceberg", "transition"], [file_icebergs, file_icebergs]):
                aois = gpd.read_file(file_aoi)

                if key == "iceberg":
                    aois = aois[aois.area >= 100**2]
                    print(file_icebergs)
                    print("Icebergs", len(aois))
                    print(np.mean(np.sqrt(aois.area)))
                    sizes.append(np.mean(np.sqrt(aois.area)))
                    sizes_all.append(np.sqrt(aois.area))
                    aois.to_file(file_aoi)

                    at = self.get_carra_at(file_s2, aois)
                    mean_ci = mean_confidence_interval(at, 0.9)
                    self.stats_at["mean"].append(mean_ci[0])
                    self.stats_at["std"].append(np.nanstd(at))
                    self.stats_at["ci_low"].append(mean_ci[1][0])
                    self.stats_at["ci_high"].append(mean_ci[1][1])
                    self.stats_at["p10"].append(np.nanpercentile(at, 10))
                    self.stats_at["p25"].append(np.nanpercentile(at, 25))
                    self.stats_at["p50"].append(np.nanmedian(at))
                    self.stats_at["p75"].append(np.nanpercentile(at, 75))
                    self.stats_at["p90"].append(np.nanpercentile(at, 90))
                    self.stats_at["min"].append(np.nanmin(at))
                    self.stats_at["max"].append(np.nanmax(at))

                aois = aois.buffer(buffer, cap_style=3).difference(aois.make_valid()) if buffer > 0 else aois
                data = np.float16(self.read_in_aoi(file_s2, aois, "iceberg"))
                self.data_all[file_icebergs].append(data)
                if data is None:
                    continue
                if key == "iceberg":
                    data = data.flatten()
                    data = data[~np.isnan(data)]
                    self.data_icebergs[file_icebergs] = data
                elif key == "transition":
                    self.data_neighborhoods[file_icebergs] = data.flatten()
                else:
                    self.data_ocean[file_icebergs] = data.flatten()
        n = [len(x) for x in self.data_icebergs.values()]
        print("Mean reference area:", mean_confidence_interval(np.float32(sizes)))
        print("Std sizes", np.std(sizes))
        print("Mean sizes:", sizes)
        print("Max size:", np.max(sizes_all))
        print("Min size:", np.min(sizes_all))
        print("N iceberg samples per SZA", np.mean(n))
        print("Std n iceberg samples", np.std(n))

    def compare_iceberg_areas(self, files_icebergs, files_s2, threshold, thresholds_sza=None):
        thresholds = [threshold] * len(files_s2) if thresholds_sza is None else thresholds_sza
        stats = {file: self.detect_icebergs(file, gpd.read_file(file_icebergs), threshold, os.path.dirname(file_icebergs)) for file, file_icebergs, threshold in zip(files_s2, files_icebergs, thresholds)}
        stats_concat = dict(mean=np.zeros(len(stats)), lower_ci=np.zeros(len(stats)), upper_ci=np.zeros(len(stats)), p25=np.zeros(len(stats)), p75=np.zeros(len(stats)), std=np.zeros(len(stats)))
        for i, (_, values) in enumerate(stats.items()):
            for stat, value in values.items():
                stats_concat[stat][i] = value
        return stats_concat

    def detect_icebergs(self, file, icebergs, threshold, dir_out):
        with rio.open(file) as src:
            data, transform = mask(src, list(icebergs.buffer(self.buffer_distance).geometry), indexes=1, nodata=0, crop=True)  # read in buffered AOI
            meta = src.meta
        data = np.float32(data) * 1e-4
        data[data == 0] = np.nan
        iceberg_polygons, _ = to_polygons(np.int8(data >= threshold), transform, meta["crs"])
        iceberg_polygons.geometry = iceberg_polygons.geometry.apply(lambda p: close_holes(p)).make_valid()  # fill holes
        for iceberg_reference in icebergs.geometry:
            aoi = iceberg_reference.buffer(self.buffer_distance)
            iceberg_detected = iceberg_polygons.clip(aoi)
            largest = iceberg_detected.iloc[np.argmax(iceberg_detected.area)]
            gdf = gpd.GeoDataFrame(geometry=[largest.geometry], crs=iceberg_polygons.crs)
            iceberg_polygons.loc[largest.name, "geometry"] = gdf.dissolve().geometry.iloc[0]  # there might be fragments detected, keep only the largest target iceberg
            iceberg_polygons.loc[largest.name, "area_reference"] = iceberg_reference.area
        iceberg_polygons = iceberg_polygons[~np.isnan(iceberg_polygons.area_reference)]
        iceberg_polygons.to_file(os.path.join(dir_out, "{}.gpkg".format(os.path.basename(file).split(".")[0])))
        proportion = np.divide(np.float32(iceberg_polygons.area), np.float32(iceberg_polygons["area_reference"]))  # buffer reference icebergs by 10 m to cover edge
        mean_ci = mean_confidence_interval(proportion)
        return {"mean": mean_ci[0], "lower_ci": mean_ci[1][0], "upper_ci": mean_ci[1][1], "p25": np.nanpercentile(proportion, 25), "p75": np.nanpercentile(proportion, 75), "std": np.nanstd(proportion)}
    
    def get_carra_at(self, file_s2, aoi):
        date_s2 = os.path.basename(file_s2).split("_")[2]
        file_carra = os.path.join(self.dir_carra, "west_domain_surface_or_atmosphere_analysis_2m_temperature_YYYY_YYYY_MM_MM_DD_DD_*_25WER_.tif")
        file_carra = file_carra.replace("YYYY", date_s2[:4]).replace("MM", str(int(date_s2[4:6]))).replace("DD", str(int(date_s2[6:8])))
        files_carra = glob(file_carra)
        y, m, d, t, minutes = int(date_s2[:4]), int(date_s2[4:6]), int(date_s2[6:8]), int(date_s2[9:11]), int(date_s2[11:13])
        date_s2 = datetime(y, m, d, t, minutes)
        diff = []
        for file in files_carra:
            diff.append(np.abs((date_s2 - datetime(y, m, d, int(os.path.basename(file).split("_")[-3]), 0)).total_seconds()))
        with rio.open(files_carra[np.argmin(diff)]) as src:
            at, _ = mask(src, list(aoi.to_crs(src.crs).geometry), crop=True, all_touched=True, nodata=np.nan, indexes=1)
        return at

    def plot_reflectance_time_series(self):
        stats_iceberg = self.calculate_stats(self.data_icebergs)
        stats_neighborhood = self.calculate_stats(self.data_neighborhoods)
        values_all_icebergs = np.float32(np.hstack(list(self.data_icebergs.values())))
        values_all_neighborhoods = np.float32(np.hstack(list(self.data_neighborhoods.values())))
        print("Mean iceberg reflectance overall", np.mean(values_all_icebergs))
        print("Std iceberg reflectance overall", np.std(values_all_icebergs))
        print("Mean neighborhoods reflectance overall", np.nanmean(values_all_neighborhoods))
        print("Std neighborhoods reflectance overall", np.nanstd(values_all_neighborhoods))
        print("Mean air temperature all dates", np.mean(self.stats_at["mean"]))
        print("Std air temperature all dates", np.std(self.stats_at["mean"]))
        print("Max air temperature all dates", np.max(self.stats_at["mean"]))
        print("Min air temperature all dates", np.min(self.stats_at["mean"]))
        print("Mean air temperature", self.stats_at["mean"])
        print("Std air temperature", self.stats_at["std"])
        print("Min air temperature", self.stats_at["min"])
        print("Max air temperature", self.stats_at["max"])
        plot = Plot(dir_plot, get_parameter("DPI_PLOTS_MEDIUM"), get_color("iceberg"))
        plot.fig, plot.axes = plt.subplots(3, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [1, 1, 0.5]})
        theta_labels = self.theta_labels.copy()
        axes = plot.axes.flatten()
        colors = [get_color("iceberg"), get_color("background"), get_color("metocean")]  #get_color("background"), 
        labels = ["Icebergs", "Neighborhoods", "2 m air temperature"]
        for ax, color, label, stats in zip([axes[0], axes[1], axes[2]], colors, labels, [stats_iceberg, stats_neighborhood, self.stats_at]):
            xs = np.float32(self.thetas)
            ax.plot(xs, np.repeat(np.mean(stats["mean"]), len(xs)), linestyle="dashed", color=color, linewidth=1, label="Global mean")
            ax.plot(xs, stats["mean"], linewidth=2.5, color=color, label="Mean")
            ax.fill_between(xs, stats["p10"], stats["p90"], color=color, alpha=0.3, label="P10-P90")
            ax.fill_between(xs, stats["p25"], stats["p75"], color=color, alpha=0.3, label="P25-P75")            
            ax.set_xticks(xs)
            if "temperature" in label:
                ax.set_ylabel("Air temperature (°C)")
            else:
                ax.set_ylabel(r"$\rho_{NIR}$")
            plot.minimalistic_layout(ax)
            ax.set_xticklabels(theta_labels)
        plot.axes[-1].set_xticks(xs)
        plot.axes[-1].set_xticklabels(theta_labels)
        plot.axes[-1].set_xlabel(THETA)
        plot.axes[0].set_ylim(0, 1.1)
        plot.axes[1].set_ylim(0, 0.1)
        plot.axes[0].set_title("a\nIcebergs")
        plot.axes[1].set_title("b\nNeighborhoods")
        plot.axes[2].set_title("c\nAir temperature")
        for ax in plot.axes:
            plot.minimalistic_layout(ax)
        plot.save_figure("iceberg_oecan_reflectance_sza")

    def plot_skewness(self):
        plot = Plot(dir_plot, get_parameter("DPI_PLOTS_MEDIUM"), get_color("iceberg"))
        plot.create_figure(2, 1, (6, 3.5))
        skewness = [skew(list(np.float32(data))) for data in self.data_icebergs.values()]
        std_dev = [np.std(list(np.float32(data))) for data in self.data_icebergs.values()]
        theta_labels = self.theta_labels.copy()
        theta = [float(t) for t in self.thetas]
        plot.axes[0].plot(theta, std_dev, linewidth=2.5, color=get_color("iceberg"))
        plot.axes[0].set_ylabel("Standard deviation")
        plot.axes[0].set_title("a")
        plot.axes[0].set_yticks([0, 0.1, 0.2, 0.3])
        plot.axes[1].plot(theta, skewness, linewidth=2.5, color=get_color("iceberg")) 
        plot.axes[1].set_yticks([-1, -0.5, 0, 0.5, 1])
        plot.axes[1].set_ylabel("Skewness")
        plot.axes[1].set_ylim(-1, 1.3)
        plot.axes[1].set_title("b")
        for ax in plot.axes:
            ax.set_xlabel(THETA)
            ax.set_xticks(theta)
            ax.set_xticklabels(theta_labels)
        plot.axes[0].set_xticks([])
        plot.axes[0].set_xticklabels([])
        plot.axes[0].set_xlabel("")
        for ax in plot.axes:
            plot.minimalistic_layout(ax)
        plot.save_figure("standard_deviation_skewness_sza")

    def plot_standardized_area_error(self, files_icebergs, files_s2):
        szas = np.float32([float(sza) for sza in self.thetas])
        proportion_stats = self.compare_iceberg_areas(files_icebergs, files_s2, BEST_THRESHOLD_FROM_CALIBRATION)
        proportion_mean = proportion_stats["mean"] * 100
        proportion_p25 = proportion_stats["p25"] * 100
        proportion_p75 = proportion_stats["p75"] * 100
        proportion_mean_interp, szas_interp = self.interpolate_error(szas, proportion_mean)
        proportion_p25_interp, szas_interp = self.interpolate_error(szas, proportion_p25)
        proportion_p75_interp, szas_interp = self.interpolate_error(szas, proportion_p75)
        re = proportion_mean_interp - 100
        error_standardized = self.standardize_error(szas_interp, re, MEAN_ERROR_FROM_CALIBRATION).flatten()
        p25_standardized, p75_standardized = self.standardize_error(szas_interp, proportion_p25_interp, P25_ERROR_FROM_CALIBRATION).ravel(), self.standardize_error(szas_interp, proportion_p75_interp, P75_ERROR_FROM_CALIBRATION).ravel()
        plot = Plot(dir_plot, get_parameter("DPI_PLOTS_MEDIUM"), get_color("iceberg"))
        plot.create_figure(1, 1, (6, 4.5))
        plot.axes.plot(szas_interp, np.zeros(len(szas_interp)), linestyle="dotted", color="black")
        plot.axes.plot(szas, proportion_mean - 100, color="black", linewidth=2, alpha=0.2, label=r"$RE_{\theta}$ mean")
        #p25 = error_standardized + ((proportion_p25_interp - 100) - re)
        #p75 = error_standardized + ((proportion_p75_interp - 100) - re)
        plot.axes.fill_between(
            szas_interp, 
            p25_standardized,
            p75_standardized,
            color=get_color("iceberg"),
            alpha=0.25,
            label=r"$SRE_{\theta}$ Q1-Q3"
            )
        plot.axes.plot(szas_interp, error_standardized, color=get_color("iceberg"), linewidth=3.5, label=r"$SRE_{\theta}$ mean")
        twinx = plot.axes.twinx()
        spread = p75_standardized - p25_standardized
        twinx.plot(szas_interp, spread, color=plot.color, linestyle="dashed", linewidth=2, label="\n".join([r"$SRE_{\theta}$ interquartile spread", "(right y-axis)"]))
        twinx.set_ylim(0, np.max(spread) + 2)
        twinx.set_yticklabels([int(label) for label in twinx.get_yticks()])
        twinx.set_ylabel(r"$SRE_{\theta}$ interquartile spread (% points)")
        plot.minimalistic_layout(twinx)
        plot.axes.set_ylabel("Relative error (%)")
        plot.axes.set_xlabel(THETA)
        plot.axes.set_xticks(szas_interp[::5])
        plot.axes.set_xticklabels([str(int(sza)) for sza in szas_interp[::5]])
        plot.axes.annotate("Regime 1", xy=(55, -32), xytext=(55, -29), ha="center", va="bottom", xycoords="data", arrowprops=dict(arrowstyle="-[, widthB=8.2, lengthB=0.5", lw=1))
        plot.axes.annotate("Regime 2", xy=(68.5, -32), xytext=(68.5, -29), ha="center", va="bottom", xycoords="data", arrowprops=dict(arrowstyle="-[, widthB=3.1, lengthB=0.5", lw=1))
        plot.axes.annotate("Regime 3", xy=(76.5, -32), xytext=(76.5, -29), ha="center", va="bottom", xycoords="data", arrowprops=dict(arrowstyle="-[, widthB=3.6, lengthB=0.5", lw=1))
        plot.axes.set_ylim(-35, 35)
        plot.axes.set_yticks(np.arange(plot.axes.get_ylim()[0], plot.axes.get_ylim()[1] + 1, step=5))
        handles, labels = plot.axes.get_legend_handles_labels()
        handles_twinx, labels_twinx = twinx.get_legend_handles_labels()
        plot.axes.legend(np.hstack([handles[::-1], handles_twinx]), np.hstack([labels[::-1], labels_twinx]), fontsize=9, loc="upper left")
        plot.minimalistic_layout(plot.axes)
        plot.save_figure(f"relative_error_sza_threshold_{BEST_THRESHOLD_FROM_CALIBRATION}")
        self.plot_error_spread(szas_interp, p25_standardized, p75_standardized)

    def draw_brace(ax, xspan, yy, text):
        """Draws an annotated brace on the axes. https://stackoverflow.com/questions/18386210/annotating-ranges-of-data"""
        xmin, xmax = xspan
        xspan = xmax - xmin
        ax_xmin, ax_xmax = ax.get_xlim()
        xax_span = ax_xmax - ax_xmin

        ymin, ymax = ax.get_ylim()
        yspan = ymax - ymin
        resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
        beta = 300./xax_span # the higher this is, the smaller the radius

        x = np.linspace(xmin, xmax, resolution)
        x_half = x[:int(resolution/2)+1]
        y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                        + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
        y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
        y = yy + (.05*y - .01)*yspan # adjust vertical position
        ax.autoscale(False)
        ax.plot(x, y, color="black", lw=1)
        ax.text((xmax+xmin)/2., yy+.07*yspan, text, ha="center", va="bottom")

    def plot_error_spread(self, szas_interp, p25, p75):
        spread = p75 - p25
        plot = Plot(dir_plot, get_parameter("DPI_PLOTS_MEDIUM"), get_color("iceberg"))
        plot.create_figure(1, 1, (4.5, 3))
        plot.axes.plot(szas_interp, spread, color=plot.color, linewidth=3.5)
        plot.axes.set_xlabel(THETA)
        plot.axes.set_ylabel("SRE spread (% points)")
        plot.minimalistic_layout(plot.axes)
        plot.save_figure(f"relative_error_spread_sza_threshold_{BEST_THRESHOLD_FROM_CALIBRATION}")

    def interpolate_error(self, szas, proportion):
        interpolator = interp1d(szas, proportion)
        szas_for_interp = np.arange(np.min(szas), np.max(szas) + 1)
        interpolated = interpolator(szas_for_interp)
        averaged = np.zeros(len(interpolated))
        for i, value in enumerate(interpolated):
            averaged[i] = np.mean(interpolated[int(np.clip(i - 2, 0, np.inf)) : int(np.clip(i + 3, 0, len(interpolated)))])
        return averaged, szas_for_interp

    def plot_histograms(self):
        plot = Plot(dir_plot, get_parameter("DPI_PLOTS_MEDIUM"), get_color("iceberg"))
        plot.create_figure(3, int(np.round(len(self.data_icebergs) * (1 / 3))), (9, 6))
        plot.axes[-1, -1].axis("off")
        #plot.axes[-1, -2].axis("off")
        for ax in plot.axes.flatten():
            ax.set_ylim(0, 0.2)
            ax.set_xlim(0, 1)
        for i, (ax, title, key) in enumerate(zip(
            plot.axes.flatten(),
            self.theta_labels,
            self.data_icebergs.keys()
        )):
            for data, color in zip([self.data_icebergs[key], self.data_neighborhoods[key]], [get_color("iceberg"), get_color("background")]):
                plot.axes = ax
                plot.color = color
                plot.plot_histogram(
                    np.float32(list(data)),
                    r"$\rho_{NIR}$",
                    f"{title}°",
                    bins=40,
                    percentage=True,
                    alpha=0.6
                )
                print("Median", np.nanmedian(list(data)))
            ymax = plot.axes.get_ylim()[1]
            plot.axes.plot([BEST_THRESHOLD_FROM_CALIBRATION, BEST_THRESHOLD_FROM_CALIBRATION], [0, ymax], linestyle="--", linewidth=0.5, color="black")
            plot.axes.set_ylim(0, 0.12)
            plot.minimalistic_layout(plot.axes)
        plot.fig.legend(
            [plot.axes.get_children()[0], plot.axes.get_children()[40], plot.axes.get_children()[80]], 
            ["Icebergs", "Neighborhoods", "Calibrated\nthreshold"],
            loc="upper left",
            bbox_to_anchor=(0.8, 0.3),
            ncol=1)
        plot.save_figure(f"iceberg_ocean_histograms_by_sza")

    def read_s2(self, files_s2, which):
        data_s2 = {self.date_from_s2_file(file_s2): self._read_in_aoi(file_s2, which) if (len(gpd.read_file(file_aoi)) > 0 if which != "ocean_extreme" else len(gpd.read_file(file_aoi)) > 1) else None for file_s2, file_aoi in zip(files_s2, self.files_out[which])}
        delete = []
        for i, (key, value) in enumerate(data_s2.items()):
            if value is None:
                delete.append(key)
                self.files_out[which][i] = ""
        for key in delete:
            del data_s2[key]
        return data_s2  

    def get_szas(self, data_s2, which):
        dates = np.array(list(data_s2.keys()))
        szas = np.array([calc_sza(
            dict(startDate=date.strftime(get_constant("FORMAT_DATE_TIME")), 
            geometry=gpd.read_file(file_aoi).to_crs("EPSG:4326").geometry.iloc[0].centroid)
            ) for date, file_aoi in zip(dates, self.files_out[which])])
        return szas, np.array([date.strftime(get_constant("FORMAT_DATE")) for date in dates])

    def standardize_error(self, szas, re, reference):
        return re - re[np.argwhere(szas == 56)] + reference  # 56 degrees SZA (Svalbard calibration)

    @staticmethod
    def percentile(data, p):
        return np.float32([np.nanpercentile(datum, p) for datum in data])

    @staticmethod
    def calculate_stats(data):
        stats = dict(mean=[], ci_low=[], ci_high=[], p5=[], p10=[], p25=[], p50=[], p75=[], p90=[], p95=[], std=[])
        for values in data.values():
            values = np.float32(list(values))
            mean_ci = mean_confidence_interval(values, 0.9)
            stats["mean"].append(mean_ci[0])
            stats["ci_low"].append(mean_ci[1][0])
            stats["ci_high"].append(mean_ci[1][1])
            stats["std"].append(np.nanstd(values))
            stats["p5"].append(np.nanpercentile(values, 5))
            stats["p10"].append(np.nanpercentile(values, 10))
            stats["p25"].append(np.nanpercentile(values, 25))
            stats["p50"].append(np.nanpercentile(values, 50))
            stats["p75"].append(np.nanpercentile(values, 75))
            stats["p90"].append(np.nanpercentile(values, 90))
            stats["p95"].append(np.nanpercentile(values, 95))
        return stats

    @staticmethod
    def date_from_s2_file(file_s2):
        date = os.path.basename(file_s2).split("_")[2]
        return datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), int(date[9:11]), int(date[11:13]), int(date[-2:]))

    def get_file_aoi(self, file_s2, which):
        which = "icebergs" if which == "iceberg" else which
        return os.path.join(self.dir_aois, which, "{}.gpkg".format(os.path.basename(file_s2).split("_")[2]))
    
    def _read_in_aoi(self, file_s2, which):
        return self.read_in_aoi(file_s2, gpd.read_file(self.get_file_aoi(file_s2, which)), which)

    @staticmethod
    def read_in_aoi(file_s2, aoi, which):
        with rio.open(file_s2) as src:
            try:
                data_s2, _ = mask(src, list(aoi.geometry), nodata=0, crop=True, indexes=[1, 5])
            except ValueError:
                return
            data_s2 = data_s2.astype(np.float32, copy=False)
            data_s2[0, data_s2[-1] > 5] = np.nan
            data_s2 = data_s2[0]  # B8
            data_s2 *= 1e-4
            data_s2[data_s2 == 0] = np.nan
        return data_s2
        
    @staticmethod
    def acquisition_szas(files_s2):
        szas = dict.fromkeys(files_s2)
        for file_s2 in files_s2:
            with rio.open(file_s2) as src:
                bounds = src.bounds
                centroid = gpd.GeoDataFrame(geometry=[box(bounds.left, bounds.bottom, bounds.right, bounds.top).centroid], crs=src.crs)
            date = os.path.basename(file_s2).split("_")[2]
            acq = dict(geometry=centroid.to_crs("EPSG:4326").geometry.iloc[0], startDate="T".join(["-".join([date[:4], date[4:6], date[6:8]]), ":".join([date[9:11], date[11:13], date[13:15]])]))
            szas[file_s2] = calc_sza(acq)
        argsorted = np.argsort(list(szas.values()))
        for file, sza in zip(np.array(list(szas.keys()))[argsorted], np.float32(list(szas.values()))[argsorted]):
            print(os.path.basename(file), "  SZA:  ", int(np.round(sza)))


if __name__ == "__main__":
    thetas = [str(angle) for angle in [45, 47, 49, 52, 56, 58, 60, 62, 64, 67, 70, 73, 75, 81]]
    files_s2 = [
        os.path.join(dir_s2, basename) for basename in [
            "S2B_MSIL1C_20200704T140739_N0209_R053_T25WER_20200704T142444_pB2.09.jp2",  # 45
            "S2A_MSIL1C_20160720T141012_N0204_R053_T25WER_20160720T141008_pB2.04.jp2",  # 47
            "S2A_MSIL1C_20180727T140021_N0206_R010_T25WER_20180727T160840_pB2.06.jp2",  # 49
            "S2B_MSIL1C_20170809T141009_N0205_R053_T25WER_20170809T141242_pB2.05.jp2",  # 52  
            "S2B_MSIL1C_20200820T135739_N0209_R010_T25WER_20200820T161028_pB2.09.jp2",  # 56
            "S2A_MSIL1C_20200825T140021_N0209_R010_T25WER_20200825T174728_pB2.09.jp2",  # 58  
            "S2A_MSIL1C_20170831T140011_N0205_R010_T25WER_20170831T140014_pB2.05.jp2",  # 60  
            "S2A_MSIL1C_20170907T135031_N0205_R110_T25WER_20170907T135027_pB2.05.jp2",  # 62  
            "S2A_MSIL1C_20170913T141001_N0205_R053_T25WER_20170913T141002_pB2.05.jp2",  # 64
            "S2B_MSIL1C_20180920T135839_N0206_R010_T25WER_20180920T175206_pB2.06.jp2",  # 67  
            "S2B_MSIL1C_20190928T140949_N0208_R053_T25WER_20190928T142401_pB2.08.jp2",  # 70
            "S2A_MSIL1C_20211006T135041_N0301_R110_T25WER_20211006T141320_pB3.01.jp2",  # 73  
            "S2B_MSIL1C_20201009T140009_N0209_R010_T25WER_20201009T142925_pB2.09.jp2",  # 75  
            "S2B_MSIL1C_20201026T135159_N0209_R110_T25WER_20201026T141203_pB2.09.jp2",  # 81
        ]
    ]
    files_icebergs = [
        os.path.join("/home/henrik/Output/icebergs/s2_sza_analysis/aois/greenland", "{}_icebergs.gpkg".format(os.path.basename(file).split("_")[2])) for file in files_s2
    ]
    
    s2sza = S2SZA(thetas, dir_main, dir_aois, dir_carra)
    s2sza.acquisition_szas(files_s2)
    s2sza.read_data(files_icebergs, files_s2)
    s2sza.plot_histograms()
    s2sza.plot_skewness()
    s2sza.plot_reflectance_time_series()
    s2sza.plot_standardized_area_error(files_icebergs, files_s2)
