library(Seurat)
library(dplyr)
library(Matrix)
path = "./GSE140228"  

# Load processed 10x matrix
mat = ReadMtx(
  mtx = file.path(path, "GSE140228_UMI_counts_Droplet.mtx"),
  features = file.path(path, "GSE140228_UMI_counts_Droplet_genes.tsv"),
  cells = file.path(path, "GSE140228_UMI_counts_Droplet_barcodes.tsv"),
  skip.feature  = 1,
  feature.column = 2
)

obj_all = CreateSeuratObject(
  counts = mat,
  min.cells = 3,
  min.features = 200
)

obj_all

# Load metadata
cellinfo = read.delim(file.path(path, "GSE140228_UMI_counts_Droplet_cellinfo.tsv"),
                       header = TRUE, sep = "\t", stringsAsFactors = FALSE)

head(cellinfo)
colnames(cellinfo)


# Attach metadata to Seurat object
rownames(cellinfo) = cellinfo$Barcode
cellinfo_use = cellinfo[colnames(obj_all), ]
obj_all = AddMetaData(obj_all, metadata = cellinfo_use)
head(obj_all@meta.data)

table(obj_all$Tissue)
table(obj_all$Tissue_sub)

# T cell subset (author annotations)
obj_T = subset(obj_all, subset = celltype_global == "Lymphoid-T")

table(obj_T$Tissue)
table(obj_T$Tissue_sub)

# Pseudo-bulk function
pseudo_by_tissue <- function(obj, tissue){
  sub <- subset(obj, subset = Tissue_sub == tissue)
  mat <- GetAssayData(sub, slot = "counts")
  pb <- Matrix::rowSums(mat)
  return(pb)
}

# Aggregate T cells by tissue
pb_blood = pseudo_by_tissue(obj_T, "Blood")
pb_tumorcore = pseudo_by_tissue(obj_T, "TumorCore")
pb_tumoredge = pseudo_by_tissue(obj_T, "TumorEdge")

# Combined pseudo-bulk matrix
pseudo_mat = cbind(Blood = pb_blood,
                    TumorCore = pb_tumorcore,
                    TumorEdge = pb_tumoredge)

head(pseudo_mat)

# Save for downstream metabolic modeling
write.csv(pseudo_mat, "pseudo_bulk_counts_T_cells.csv")
