#!/opt/homebrew/bin/Rscript
.libPaths(c("~/R/library", .libPaths()))
library(ggplot2)

# ======================================================================
# USER CONFIG — update these paths to match your local setup
# data_dir: directory containing CD3 data CSVs
# output_dir: where figures will be saved
# ======================================================================
data_dir   <- "../data/cd3"
output_dir <- "./output"

outdir <- output_dir

freq <- read.csv(file.path(data_dir, "cd3_mc20_frequencies.csv"),
                 row.names = 1, check.names = FALSE)

# Group assignment
get_group <- function(s) {
  if (grepl("_HC", s)) return("HC")
  if (grepl("_PrEP", s)) return("PrEP")
  if (grepl("_HIV_W48", s)) return("HIV W48")
  if (grepl("_HIV_W0", s)) return("HIV W0")
  return("Unknown")
}
groups <- sapply(rownames(freq), get_group)

prep_data <- freq[groups == "PrEP", ]
hc_data   <- freq[groups == "HC", ]
w0_data   <- freq[groups == "HIV W0", ]

mc_names <- colnames(freq)

# ── Function to make volcano ──
make_volcano <- function(group_a, group_b, label_a, label_b, filename) {
  results <- data.frame(MC = mc_names, diff = NA, pval = NA, padj = NA)

  for (i in seq_along(mc_names)) {
    mc <- mc_names[i]
    a <- group_a[, mc]
    b <- group_b[, mc]
    results$diff[i] <- mean(a) - mean(b)
    wt <- wilcox.test(a, b)
    results$pval[i] <- wt$p.value
  }

  results$padj <- p.adjust(results$pval, method = "BH")
  results$neg_log_padj <- -log10(results$padj)
  results$sig <- ifelse(results$padj < 0.05 & results$diff > 0, "enriched",
                        ifelse(results$padj < 0.05 & results$diff < 0, "depleted", "ns"))

  sig_colors <- c("enriched" = "#008000", "depleted" = "#CC0000", "ns" = "grey60")

  p <- ggplot(results, aes(x = diff, y = neg_log_padj, color = sig)) +
    geom_point(size = 4, alpha = 0.85) +
    geom_text(data = results[results$padj < 0.05, ],
              aes(label = MC), fontface = "bold", size = 5,
              vjust = -1, show.legend = FALSE) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "salmon", linewidth = 0.7) +
    geom_vline(xintercept = 0, color = "grey30", linewidth = 0.4) +
    scale_color_manual(values = sig_colors) +
    annotate("text", x = min(results$diff) * 0.9, y = -log10(0.05) + 0.08,
             label = "padj = 0.05", color = "salmon", size = 4, hjust = 0) +
    labs(title = paste0(label_a, " vs ", label_b, ": Metacluster Volcano Plot"),
         subtitle = paste0("Green = ", label_a, "-enriched, Red = ", label_a, "-depleted"),
         x = paste0("Frequency difference (", label_a, " \u2212 ", label_b, ", %)"),
         y = expression(-log[10](adjusted~p-value))) +
    theme_classic(base_size = 16) +
    theme(plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
          plot.subtitle = element_text(face = "bold", size = 14, hjust = 0.5),
          axis.title = element_text(face = "bold"),
          legend.position = "none")

  ggsave(file.path(outdir, filename), p, width = 10, height = 8, dpi = 300)
  cat(sprintf("Saved %s\n", filename))

  # Print significant results
  sig_res <- results[results$padj < 0.05, c("MC","diff","padj")]
  sig_res <- sig_res[order(sig_res$diff), ]
  cat(sprintf("\nSignificant MCs (%s vs %s):\n", label_a, label_b))
  for (r in 1:nrow(sig_res)) {
    cat(sprintf("  %s: diff=%.1f%%, padj=%.4f\n", sig_res$MC[r], sig_res$diff[r], sig_res$padj[r]))
  }
  cat("\n")
}

# ── PrEP vs HC ──
make_volcano(prep_data, hc_data, "PrEP", "HC", "PrEP_vs_HC_Volcano.png")

# ── PrEP vs HIV W0 ──
make_volcano(prep_data, w0_data, "PrEP", "HIV W0", "PrEP_vs_HIV_W0_Volcano.png")

cat("Both volcano plots done!\n")
