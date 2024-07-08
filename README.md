# Iceberg area retrieval from Sentinel-2 data at varying solar angles

## About the repository
This repository contains the Python code underlying the peer-reviewed paper titled "Impact of varying solar angles on Arctic iceberg area retrieval from Sentinel-2 near-infrared data". In the paper we calibrated a near-infrared reflectance threshold for the iceberg area retrieval from Sentinel-2 data. Further, we quantified the error variation of the iceberg area retrieval with the solar angle. We recommend reading the paper for details. In this repository we share the code that was written to conduct the analysis. This repository contains code that was written to conduct the experiments. The purpose is to publish the calculations implemented in the code along with the paper. This repository is no standalone software and it is not an installable Python package. Furthermore, it does not contain the data needed to run the analysis. Please reach out if you seek the data used in the study. 

## Summary
The separation between ice and ocean pixels in near-infrared data appears straight-forward initially. Many have used constant reflectance thresholds for this purpose. In our paper we propose to use the following rule for detecting and delineating icebergs from top-of-atmosphere Sentinel-2 near-infrared (B08) reflectance (scaled to 0-1 range) data:

$$iceberg = B08 >= 0.12$$

Connected pixels are considered one object. The individual iceberg area is then the sum over the area covered by the connected pixels. In the paper we examined the effects of varying solar zenith angles on the error in the area retrieval, using the reflectance threshold. 

We recommend limiting the iceberg area retrieval from these data to solar zenith angles below 65°, which is 5° lower than the broad recommendation by the [European Space Agency](https://scihub.copernicus.eu/news/News00610). The error in the iceberg area retrieval is consistent up to 65°, staying approximately between an overestimated +6% and an underestimated -6%. The specific error margins apply to the threshold used here, but we expect the varying solar illuminations to challenge the iceberg area retrieval also when using other algorithms. 

## Background
Iceberg areas are important for studies on freshwater and nutrient fluxes in glacial fjords, and for iceberg drift and deterioration modelling. Satellite remote sensing encompasses valuable tools and datasets to calculate iceberg areas. Synthetic aperture radar (SAR) and optical satellite data are of particular relevance. We generally aim to use iceberg area retrievals from SAR data in conjunction with optical data to study Arctic iceberg populations. The snapshot character of optical satellite acquisitions on cloud-free days is suitable for verifying retrievals from SAR data, but cannot facilitate a consistent time series. To use optical data for verification we have to understand errors in the iceberg area retrieval from optical data first, which was the objective of the paper. 

## Contact
Use the contact details of the first author shared in the publication referenced above.
