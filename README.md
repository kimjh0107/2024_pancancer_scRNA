## Pan-cancer gene set discovery via scRNA-seq for optimal deep learning based downstream tasks

Jong Hyun Kim<sup>1+</sup>, BS;
Jongseong Jang<sup>1∗</sup>, PhD

<sup>1</sup>LG AI Research, Seoul, South Korea
> **Abstract**: The application of deep learning to transcriptomics data has led to significant advances in cancer research. However, the high dimensionality and complexity of RNA sequencing (RNA-seq) data pose significant challenges in pan-cancer studies. This study aims to leverage single-cell RNA sequencing (scRNA-seq) data to improve feature selection in pan-cancer studies, hypothesizing that gene sets derived from scRNA-seq will outperform traditional bulk RNA-seq in predictive modeling. We analyzed scRNA-seq data from 181 tumor biopsies across 13 cancer types. High-dimensional weighted gene co-expression network analysis (hdWGCNA) was used to identify relevant gene sets, which were further refined with XGBoost for feature selection. These gene sets were applied to downstream tasks using TCGA pan-cancer RNA-seq data and compared to six reference gene sets and OncoKB database gene sets evaluated with deep learning models. The XGBoost-refined hdWGCNA gene set showed higher performance in most tasks, including tumor mutation burden assessment, microsatellite instability classification, mutation prediction, cancer subtyping, and grading. In particular, genes such as DPM1, BAD, and FKBP4 emerged as important pan-cancer biomarkers, with DPM1 consistently significant across tasks. This study presents a robust approach for feature selection in cancer genomics by integrating scRNA-seq data and advanced analysis techniques.

## Overview
### Our Study workflow 

![initial](https://github.com/kimjh0107/2024_pancancer_scRNA/assets/83206535/76830403-ea4a-4746-8317-74d36f2f3947)


🔐 Note: Projects that are currently in progress are set to private for confidentiality reasons.
