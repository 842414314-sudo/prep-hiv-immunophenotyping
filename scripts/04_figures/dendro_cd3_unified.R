#!/opt/homebrew/bin/Rscript
.libPaths(c("~/R/library", .libPaths()))

library(pheatmap)
library(grid)

# ======================================================================
# USER CONFIG — update these paths to match your local setup
# data_dir: directory containing CD3 data CSVs
# output_dir: where figures will be saved
# ======================================================================
data_dir   <- "../data/cd3"
output_dir <- "./output"

outdir <- output_dir

# Read frequency data
freq_df <- read.csv(file.path(data_dir, "cd3_mc20_frequencies.csv"),
                    row.names = 1)
freq_mat <- as.matrix(freq_df)

# ── Relabel: "P1_HIV_W0" → "1 W0", "P88_HC" → "88 HC" ──
relabel <- function(s) {
  s <- sub("^P", "", s)
  s <- sub("_HC_W0$", " HC", s)
  s <- sub("_PrEP_W0$", " PrEP", s)
  s <- sub("_HIV_W0$", " W0", s)
  s <- sub("_HIV_W48$", " W48", s)
  s
}

new_labels <- sapply(rownames(freq_mat), relabel)
rownames(freq_mat) <- new_labels

# ── Group assignment ──
get_group <- function(s) {
  if (grepl(" W0$", s)) return("HIV W0")
  if (grepl(" W48$", s)) return("HIV W48")
  if (grepl("PrEP$", s)) return("PrEP")
  return("HC")
}

groups <- sapply(new_labels, get_group)
ann_row <- data.frame(Group = groups, row.names = new_labels)

# ── Unified paper colors ──
group_colors <- c(
  "HC"      = "#008000",
  "PrEP"    = "#3E0080",
  "HIV W0"  = "#FFCD65",
  "HIV W48" = "#808000"
)
ann_colors <- list(Group = group_colors)

# ── SCI teal gradient (same as CD4/CD8 dendrograms) ──
teal_palette <- colorRampPalette(c(
  rgb(252, 253, 211, max=255),
  rgb(218, 240, 185, max=255),
  rgb(146, 212, 195, max=255),
  rgb(62, 179, 184, max=255),
  rgb(30, 128, 184, max=255),
  rgb(36, 65, 154, max=255),
  rgb(10, 31, 93, max=255)
))(100)

# ── Raw % with cap ──
breaks <- seq(0, 40, length.out = 101)
freq_mat_cap <- pmin(freq_mat, 40)

d_rows <- dist(freq_mat, method = "euclidean")
hc_rows <- hclust(d_rows, method = "ward.D2")

# Use the hclust object directly so pheatmap doesn't re-order
p <- pheatmap(freq_mat_cap,
         cluster_rows = hc_rows,
         cluster_cols = TRUE,
         annotation_row = ann_row,
         annotation_colors = ann_colors,
         annotation_legend = FALSE,
         legend = TRUE,
         annotation_names_row = FALSE,
         color = teal_palette,
         breaks = breaks,
         main = "CD3+ FlowSOM: Patient-Level Hierarchical Clustering",
         fontsize = 16,
         fontsize_row = 13,
         fontsize_col = 12,
         cellheight = 18,
         cellwidth = 18,
         border_color = "grey85",
         angle_col = 45,
         silent = TRUE)

# Bold all text
for (i in seq_along(p$grobs)) {
  if (inherits(p$grobs[[i]], "text") || inherits(p$grobs[[i]], "titleGrob")) {
    p$grobs[[i]]$gp$fontface <- "bold"
  }
  if (!is.null(p$grobs[[i]]$children)) {
    for (j in seq_along(p$grobs[[i]]$children)) {
      if (inherits(p$grobs[[i]]$children[[j]], "text")) {
        p$grobs[[i]]$children[[j]]$gp$fontface <- "bold"
      }
    }
  }
}

# Vertical version (original)
png(file.path(outdir, "CD3_Dendrogram_Unified.png"), width = 14, height = 22, units = "in", res = 300)
grid.newpage()
grid.draw(p$gtable)
dev.off()

cat("Saved CD3 dendrogram with unified colors\n")
