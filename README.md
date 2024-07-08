# Iceberg area retrieval from Sentinel-2 data at varying solar angles

## Summary
This repository contains the Python code underlying the peer-reviewed paper titled "Impact of varying solar angles on Arctic iceberg area retrieval from Sentinel-2 near-infrared data". In the paper we calibrated a near-infrared reflectance threshold for the iceberg area retrieval from Sentinel-2 data. Further, we quantified the error variation of the iceberg area retrieval with the solar angle. We recommend reading the paper for details. In this repository we share the code that was written to conduct the analysis.

The method for detecting and delineating icebergs from top-of-atmosphere Sentinel-2 near-infrared (B08) reflectance (scaled to 0-1 range) data is simple:

$$iceberg = B08 >= 0.12$$.



## Background
Iceberg areas are important for studies on freshwater and nutrient fluxes in glacial fjords, and for iceberg drift and deterioration modelling. Satellite remote sensing encompasses valuable tools and datasets to calculate iceberg areas. Synthetic aperture radar (SAR) and optical satellite data are of particular relevance. We generally aim to use iceberg area retrievals from SAR data in conjunction with optical data to study Arctic iceberg populations. The snapshot character of optical satellite acquisitions on cloud-free days is suitable for verifying retrievals from SAR data, but cannot facilitate a consistent time series. To use optical data for verification we have to understand errors in the iceberg area retrieval from optical data first, which was the objective of the paper. 

## About the repository
This repository contains code that was written to conduct the experiments. The purpose is to publish the calculations implemented in the code along with the paper. This repository is no standalone software and it is not an installable Python package. Furthermore, it does not contain the data needed to run the analysis. Please reach out if you seek the data used in the study. 

## Contact
Use the contact details of the first author shared in the publication referenced above.
