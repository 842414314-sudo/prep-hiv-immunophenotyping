#!/opt/homebrew/bin/Rscript
# ==================================================================
# CD4 / CD8 FlowSOM patient-level clustermap  —  MC = 10
# Replicates dendro_cd3_unified.R style (teal palette, square cells,
# bold text, grey85 borders). Only inputs/titles/outputs differ.
# ==================================================================
.libPaths(c("~/R/library", .libPaths()))

suppressMessages({
  library(pheatmap)
  library(grid)
})

# ======================================================================
# USER CONFIG — update these paths to match your local setup
# data_dir: root of the data/ directory (contains cd4/, cd8/ subdirs)
# output_dir: where figures will be saved
# ======================================================================
data_dir   <- "../data"
output_dir <- "./output"

indir   <- data_dir
outdir  <- output_dir

# ── Unified paper colors (SAME as CD3) ──
group_colors <- c(
  "HC"      = "#008000",
  "PrEP"    = "#3E0080",
  "HIV W0"  = "#FFCD65",
  "HIV W48" = "#808000"
)
ann_colors <- list(Group = group_colors)

# ── SCI teal gradient (SAME as CD3) ──
teal_palette <- colorRampPalette(c(
  rgb(252, 253, 211, max=255),
  rgb(218, 240, 185, max=255),
  rgb(146, 212, 195, max=255),
  rgb(62, 179, 184, max=255),
  rgb(30, 128, 184, max=255),
  rgb(36, 65, 154, max=255),
  rgb(10, 31, 93, max=255)
))(100)

# ── Relabel "P1_HIV_W0" → "1 W0", "P88_HC" → "88 HC", etc. ──
relabel <- function(s) {
  s <- sub("^P", "", s)
  s <- sub("_HIV_W0$",  " W0",   s)
  s <- sub("_HIV_W48$", " W48",  s)
  s <- sub("_PrEP$",    " PrEP", s)
  s <- sub("_HC$",      " HC",   s)
  s
}

get_group <- function(s) {
  if (grepl(" W0$",   s)) return("HIV W0")
  if (grepl(" W48$",  s)) return("HIV W48")
  if (grepl(" PrEP$", s)) return("PrEP")
  if (grepl(" HC$",   s)) return("HC")
  return("Unknown")
}

make_dendro <- function(subset) {
  subset_lower <- tolower(subset)
  csv_path <- file.path(indir, subset_lower, sprintf("%s_mc10_frequencies.csv", subset_lower))
  freq_df <- read.csv(csv_path, row.names = 1, check.names = FALSE)
  freq_mat <- as.matrix(freq_df)

  new_labels <- sapply(rownames(freq_mat), relabel)
  rownames(freq_mat) <- new_labels

  groups <- sapply(new_labels, get_group)
  ann_row <- data.frame(Group = groups, row.names = new_labels)

  breaks <- seq(0, 40, length.out = 101)
  freq_mat_cap <- pmin(freq_mat, 40)

  d_rows  <- dist(freq_mat, method = "euclidean")
  hc_rows <- hclust(d_rows, method = "ward.D2")

  p <- pheatmap(
    freq_mat_cap,
    cluster_rows          = hc_rows,
    cluster_cols          = TRUE,
    annotation_row        = ann_row,
    annotation_colors     = ann_colors,
    annotation_legend     = FALSE,
    annotation_names_row  = FALSE,
    legend                = TRUE,
    color                 = teal_palette,
    breaks                = breaks,
    main                  = sprintf("%s+ FlowSOM: Patient-Level Hierarchical Clustering",
                                    subset),
    fontsize              = 16,
    fontsize_row          = 13,
    fontsize_col          = 12,
    cellheight            = 18,
    cellwidth             = 18,
    border_color          = "grey85",
    angle_col             = 45,
    silent                = TRUE
  )

  # bold all text grobs (same as CD3)
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

  out_png <- file.path(outdir, sprintf("%s_Dendrogram_Unified.png", subset))
  out_pdf <- file.path(outdir, sprintf("%s_Dendrogram_Unified.pdf", subset))

  # 51 rows × 10 cols + annotation + dendro + title: match CD3's 14×22
  png(out_png, width = 14, height = 22, units = "in", res = 300)
  grid.newpage(); grid.draw(p$gtable); dev.off()

  pdf(out_pdf, width = 14, height = 22)
  grid.newpage(); grid.draw(p$gtable); dev.off()

  cat(sprintf("  wrote %s_Dendrogram_Unified.png/pdf\n", subset))
}

for (subset in c("CD4", "CD8")) make_dendro(subset)
cat("DONE.\n")
