library(here)
library(Seurat)
library(MetBrewer)
library(harmony)
library(doParallel)
library(foreach)
library(tidyverse)
library(hdWGCNA)
library(igraph)
library(Palo)
library(pheatmap)
library(ComplexHeatmap)
library(enrichR)
library(purrr)

set.seed(42)

# ==== Heatmap: Pathway analysis ====
data_path <- '/workspace/scRNA/data/meta/WGCNA/UMAP_hdWGCNA.csv'
db <- 'GO_Biological_Process_2023'
save_path <- '/workspace/scRNA/figure/figure2/Pathway_heatmap_kME100.pdf'

df <- read_csv(data_path) 
df <- genes[!grepl("^MT", df$gene), ]

module_colors <- df %>% select(module, color) %>% distinct()
print(module_colors)

module <- df %>%
  group_by(module) %>%
  arrange(desc(kME)) %>%
  slice_max(order_by = kME, n = 100)


module_names <- unique(module$module)
enriched_results <- map(module_names, function(module_name) {
  genes <- module %>%
    filter(module == !!module_name) %>%
    pull(gene)
  
  enriched <- enrichr(genes, db)
  enriched_data <- enriched[[db]]
  
  if (nrow(enriched_data) > 0) {  
    enriched_data$module <- module_name
    enriched_data$Q_value <- -log(enriched_data$Adjusted.P.value)
    return(enriched_data)
  } else {
    return(NULL) 
  }
}) %>% compact() %>% bind_rows() 


top5_df <- enriched_results %>% 
  group_by(module) %>% 
  top_n(5, Q_value) %>% 
  ungroup()

go_terms <- c("Negative Regulation Of Amide Metabolic Process (GO:0034249)",
              "Cholesterol Efflux (GO:0033344)",
              "Cell Junction Disassembly (GO:0150146)",
              "Humoral Immune Response Mediated By Circulating Immunoglobulin (GO:0002455)",
              "Positive Regulation Of Apoptotic Signaling Pathway (GO:2001235)",
              "Response To Fibroblast Growth Factor (GO:0071774)",
              "Aerobic Respiration (GO:0009060)",
              "Oxidative Phosphorylation (GO:0006119)",
              "Integrated Stress Response Signaling (GO:0140467)",
              "Response To Interferon-Beta (GO:0035456)",
              "mRNA Processing (GO:0006397)",
              "Positive Regulation Of Heterochromatin Formation (GO:0031453)",
              "RNA Processing (GO:0006396)",
              "Negative Regulation Of Nucleic Acid-Templated Transcription (GO:1903507)",
              "mRNA Splicing, Via Spliceosome (GO:0000398)",
              "Aerobic Electron Transport Chain (GO:0019646)",
              "Cellular Respiration (GO:0045333)",
              "Mitochondrial ATP Synthesis Coupled Electron Transport (GO:0042775)",
              "Energy Derivation By Oxidation Of Organic Compounds (GO:0015980)",
              "Mitochondrial Electron Transport, Cytochrome C To Oxygen (GO:0006123)",
              "Ribosomal Large Subunit Assembly (GO:0000027)",
              "Translational Elongation (GO:0006414)",
              "Ribosome Assembly (GO:0042255)",
              "Ribosomal Large Subunit Biogenesis (GO:0042273)",
              "Translation (GO:0006412)",
              "Cytoplasmic Translation (GO:0002181)",
              "Macromolecule Biosynthetic Process (GO:0009059)",
              "Peptide Biosynthetic Process (GO:0043043)",
              "Gene Expression (GO:0010467)",
              "Response To Unfolded Protein (GO:0006986)",
              "Protein Stabilization (GO:0050821)",
              "Chaperone-Mediated Protein Complex Assembly (GO:0051131)",
              "'De Novo' Post-Translational Protein Folding (GO:0051084)",
              "Positive Regulation Of DNA Biosynthetic Process (GO:2000573)",
              "Regulation Of Lymphocyte Migration (GO:2000401)",
              "Blood Vessel Endothelial Cell Migration (GO:0043534)",
              "Regulation Of G1/S Transition Of Mitotic Cell Cycle (GO:2000045)",
              "Positive Regulation Of Nitric Oxide Biosynthetic Process (GO:0045429)",
              "Actin Filament Polymerization (GO:0030041)",
              "Peptidyl-Tyrosine Modification (GO:0018212)",
              "Plasma Membrane Invagination (GO:0099024)",
              "Phagocytosis, Engulfment (GO:0006911)",
              "Regulation Of Lamellipodium Assembly (GO:0010591)",
              "Positive Regulation Of Leukocyte Cell-Cell Adhesion (GO:1903039)",
              "T Cell Activation (GO:0042110)",
              "Interleukin-4-Mediated Signaling Pathway (GO:0035771)",
              "Positive Regulation Of Protein Import Into Nucleus (GO:0042307)",
              "Cellular Response To Interleukin-9 (GO:0071355)",
              "Regulation Of T-helper Cell Differentiation (GO:0045622)",
              "Endoplasmic Reticulum To Golgi Vesicle-Mediated Transport (GO:0006888)",
              "COPII-coated Vesicle Budding (GO:0090114)",
              "Golgi Vesicle Transport (GO:0048193)")

filtered_top5_df <- top5_df %>%
  filter(Term %in% go_terms)
unique_terms <- unique(filtered_top5_df$Term)

result_df <- map_df(unique_terms, function(term) {
  enriched_results %>% 
    filter(Term == term) %>% 
    select(Term, module, Q_value)
})

full_df <- expand_grid(Term = unique_terms, module = module_names)
complete_df <- full_df %>%
  left_join(result_df, by = c("Term", "module")) %>%
  replace_na(list(Q_value = 0))

data_matrix <- complete_df %>%
  spread(Term, Q_value) %>%
  column_to_rownames("module") %>%
  as.matrix()

data_matrix_transposed <- t(data_matrix)
data_matrix_transposed <- t(scale(t(data_matrix_transposed)))

module_order <- c(1,2,3,4,5,6,7,8,9,10,11)
module_names <- c("Module1", 
                  "Module2",  
                  "Module3",  
                  "Module4", 
                  "Module5",
                  "Module6",  
                  "Module7",  
                  "Module8",  
                  "Module9",
                  "Module10", 
                  "Module11"
                  )

module_order_df <- data.frame(module = module_names, order = module_order)
module_order_df <- module_order_df %>% arrange(order)
data_matrix_transposed_ordered <- data_matrix_transposed[, module_order_df$module]

col_fun <- colorRampPalette(c("blue", "white", "#ff383f"))(100)

Heatmap(data_matrix_transposed_ordered,
        col = col_fun,
        cluster_columns = FALSE,
        row_names_max_width = unit(100, "cm"),
        row_names_gp = gpar(fontsize = 16, col = "black"),
        column_names_gp = gpar(fontsize = 20, col = "black"), 
        show_heatmap_legend = TRUE,
        heatmap_legend_param = list(title = "Z-score", at = c(-2, 0, 2), labels = c(-2, 0, 2), position = "bottom"),
        row_dend_width = unit(20, "mm")
        )

