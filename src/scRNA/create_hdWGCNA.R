library(here)
library(Seurat)
library(MetBrewer)
library(harmony)
library(hdWGCNA)
library(igraph)
library(doParallel)
library(foreach)
library(tidyverse)

set.seed(42)

enableWGCNAThreads(nThreads = 8)

seurat_obj <- readRDS(here('/workspace/scRNA/data/RDS/TICAtlas.rds'))
DefaultAssay(seurat_obj) <- "RNA"

genes_to_keep <- grep("^MT", rownames(seurat_obj), invert = TRUE, value = TRUE)
seurat_obj <- subset(seurat_obj, features = genes_to_keep)

subset <- SetupForWGCNA(
  seurat_obj,
  gene_select = "fraction", 
  fraction = 0.05, 
  wgcna_name = "tutorial" 
)

num_genes <- length(rownames(subset))
print(paste("Number of genes in the subset:", num_genes))

subset@misc$tutorial$wgcna_genes

DefaultAssay(subset) <- "RNA"

seurat_obj <- MetacellsByGroups(
  seurat_obj = subset,
  group.by = c("cell_type", 'patient'),
  reduction = 'pca', 
  k = 25, 
  max_shared = 10,
  ident.group = 'cell_type',
  verbose = TRUE,
)

seurat_obj <- NormalizeMetacells(seurat_obj)

seurat_obj <- SetDatExpr(
  seurat_obj,
  group_name = "Cytotoxic CD8 T cells",
  group.by='cell_type',
  assay = 'RNA', 
  slot = 'data' 
)

seurat_obj <- TestSoftPowers(
  seurat_obj,
  networkType = 'signed',
  corFnc = "bicor"
)

PlotSoftPowers(seurat_obj)

soft_power_n = 9
seurat_obj <- ConstructNetwork(
  seurat_obj, 
  soft_power=soft_power_n,
  setDatExpr=FALSE, 
  overwrite_tom = TRUE 
)

PlotDendrogram(seurat_obj, main='INH hdWGCNA Dendrogram')

modules <- GetModules(seurat_obj)
mods <- unique(modules$module)


mod_colors_df <- dplyr::select(modules, c(module, color)) %>%
  distinct %>% arrange(module)
rownames(mod_colors_df) <- mod_colors_df$module
mod_color_df <- GetModules(seurat_obj) %>%
  dplyr::select(c(module, color)) %>%
  distinct %>% arrange(module)
n_mods <- nrow(mod_color_df) - 1
new_colors <- paste0(met.brewer("Signac", n=n_mods))
seurat_obj <- ResetModuleColors(seurat_obj, new_colors)


# only subset modules_genes 
modules_genes <- rownames(modules)
seurat_obj <- subset(seurat_obj, features = modules_genes)

seurat_obj <- ScaleData(seurat_obj)

seurat_obj <- ModuleEigengenes(
  seurat_obj,
  group.by.vars="patient"
)


print('moduleconnectivity')
seurat_obj <- ModuleConnectivity(
  seurat_obj, group.by = 'cell_type', group_name = 'Cytotoxic CD8 T cells'
  )

# rename the modules
print('ResetModuleNames')
seurat_obj <- ResetModuleNames(
  seurat_obj,
  new_name = "Module"
)

seurat_obj <- RunModuleUMAP(
  seurat_obj,
  n_hubs = 10, 
  n_neighbors=15, 
  min_dist=0.1, 
)

ModuleUMAPPlot(
  seurat_obj,
  edge.alpha=0.25,
  sample_edges=TRUE,
  edge_prop=0.1, 
  label_hubs=2 ,
  keep_grey_edges=FALSE
)

umap_df <- GetModuleUMAP(seurat_obj)
write.csv(umap_df, '/workspace/scRNA/data/meta/WGCNA/UMAP_hdWGCNA.csv')

# save WGCNA processed seurat_obj
saveRDS(seurat_obj, '/workspace/scRNA/data/RDS/UMAP_hdWGCNA.rds')


