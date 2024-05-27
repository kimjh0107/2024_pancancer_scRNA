library(Seurat)
library(tidyverse)
library(Palo)
library(purrr)

set.seed(42)


seurat_obj <- readRDS('/workspace/scRNA/data/RDS/test_TICAtlas_testsoftpowers_filter_005_v2_result.rds')


# ====================== Fig.2a ======================
# DimPlot 
DefaultAssay(seurat_obj) <- "RNA"

u <- seurat_obj[['umap']]@cell.embeddings
cl <- as.character(seurat_obj$cell_type)

gg_color_hue <- function(n) {
  hues = seq(30, 500, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
pal <- gg_color_hue(length(unique(cl)))
palopal <- Palo(u,cl,pal)

color_conserved <- c(
  "#EF7F49","#7BAE00","#33B600","#00ACFC","#FF66A7",
  "#C69900","#DD8C00","#00BECB","#DA8E00","#43B500",
  "#C29A00","#82AD00","#00BC53","#A4A500","#00C083",
  "#A9A400","#00C1AA","#D874FD","#ED813E","#649BFF",
  "#00B7E7","#F265E8","#FF61CB","#AC88FF","#FC727E"
)

DimPlot(seurat_obj,
        group.by = "cell_type", 
        cols = color_conserved, 
        repel = T, 
        raster.dpi = c(300, 300),
        pt.size=0.8, 
        label.size = 3, 
        label=T) + labs(title = "") 