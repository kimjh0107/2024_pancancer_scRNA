library(Seurat)
library(tidyverse)
library(hdWGCNA)
library(igraph)

set.seed(42)


seurat_obj <- readRDS('/workspace/scRNA/data/RDS/test_TICAtlas_testsoftpowers_filter_005_v2_result.rds')

# ==== Network UMAP ====
ModuleUMAPPlot(
  seurat_obj,
  edge.alpha=0.25,
  sample_edges=TRUE,
  edge_prop=0.1, 
  label_hubs=0,
  keep_grey_edges=FALSE
)

ModuleNetworkPlot(
  seurat_obj,
  mods = 'all',
  edge.alpha=0.25,
  outdir='/workspace/scRNA/figure/figure2/',
  plot_size = c(4, 4),
  vertex.label.cex = 0.5,
  vertex.size = 6,
  )
