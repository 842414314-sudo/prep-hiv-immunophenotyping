#!/opt/homebrew/bin/Rscript
# Sys.setenv(PATH=...)  # adjust to your system
.libPaths(c("~/R/library", .libPaths()))

library(flowCore)

# ======================================================================
# USER CONFIG — update these paths to match your local setup
# FCS_DIR: root directory containing B1/, B2/, B3/ subdirectories with
#          CD3-gated FCS files exported from FlowJo
# OUTPUT_DIR: where batch-normalized FCS files will be written
# ======================================================================
FCS_DIR    <- "/Library/Claude/CD3"
OUTPUT_DIR <- path.expand("~/Desktop/normalized_CD3_v11")

set.seed(42)
cofactor <- 6000
asinh_t <- function(x) asinh(x / cofactor)
sinh_t  <- function(x) sinh(x) * cofactor

cat("============================================================\n")
cat("  V11: CD3+ Batch Normalization\n")
cat("  B1 = reference (untouched)\n")
cat("  B2 -> B1: peak alignment + median shift (P27+P33 bridges)\n")
cat("  B3 -> B1: peak alignment + median shift\n")
cat("            (B3->B2 via P33+P44, then B2->B1 via P27+P33)\n")
cat("  NOTE: 44HC is B2 (physically in B3 folder)\n")
cat("============================================================\n\n")

outdir <- OUTPUT_DIR
dir.create(outdir, showWarnings = FALSE)
dir.create(file.path(outdir, "B1"), showWarnings = FALSE)
dir.create(file.path(outdir, "B2"), showWarnings = FALSE)
dir.create(file.path(outdir, "B3"), showWarnings = FALSE)

# ── Channel definitions ───────────────────────────────────────────
channels <- c(
  CD45RA = "FJComp-cFluor V450-A",
  CCR7   = "FJComp-BV785-A",
  HLADR  = "FJComp-APC-Fire 810-A",
  CD38   = "FJComp-cFluor R720-A",
  PD1    = "FJComp-BV421-A",
  TIM3   = "FJComp-PE-A",
  CD25   = "FJComp-PE-Cy5-A",
  CD127  = "FJComp-PE-Dazzle594-A",
  FOXP3  = "FJComp-Alexa Fluor 647-A"
)

shift_markers <- c("HLADR", "CD38", "PD1", "TIM3", "CD25", "CD127", "FOXP3")

# ── Fix channel descriptions and PnR ─────────────────────────────
fix_fcs_metadata <- function(ff) {
  pd <- pData(parameters(ff))
  # Fix TIM-3 -> TIM3
  idx <- which(pd$desc == "TIM-3")
  if (length(idx) > 0) {
    pd$desc[idx] <- "TIM3"
    cat("    Fixed TIM-3 -> TIM3\n")
  }
  # Unify PnR to 99999 (match B1) for FlowJo gate compatibility
  changed <- pd$maxRange != 99999 & !is.na(pd$maxRange)
  if (any(changed)) {
    pd$maxRange[changed] <- 99999
    cat(sprintf("    Fixed $PnR -> 99999 for %d channels\n", sum(changed)))
  }
  pData(parameters(ff)) <- pd
  ff
}

# ── 2-point peak alignment ───────────────────────────────────────
align_bimodal <- function(raw_values, ref_neg, ref_pos,
                          valley_lo, valley_hi, marker_name = "") {
  asi <- asinh_t(raw_values)
  h <- hist(asi, breaks = 200, plot = FALSE)
  vr <- which(h$mids > valley_lo & h$mids < valley_hi)
  if (length(vr) > 0) {
    valley <- h$mids[vr[which.min(h$counts[vr])]]
  } else {
    valley <- (valley_lo + valley_hi) / 2
  }
  neg_vals <- asi[asi <= valley]
  pos_vals <- asi[asi > valley]
  if (length(neg_vals) < 20 || length(pos_vals) < 20) {
    cat(sprintf("    %s: too few cells, skipping\n", marker_name))
    return(raw_values)
  }
  med_neg <- median(neg_vals)
  med_pos <- median(pos_vals)
  if ((med_pos - med_neg) < 0.3) {
    cat(sprintf("    %s: peak separation too small (%.3f), skipping\n", marker_name, med_pos - med_neg))
    return(raw_values)
  }
  a <- (ref_pos - ref_neg) / (med_pos - med_neg)
  b <- ref_neg - a * med_neg
  if (a < 0.3 || a > 3.0) {
    cat(sprintf("    %s: extreme slope a=%.3f, skipping\n", marker_name, a))
    return(raw_values)
  }
  asi_corrected <- a * asi + b
  cat(sprintf("    %s: neg %.3f->%.3f, pos %.3f->%.3f (a=%.3f)\n",
              marker_name, med_neg, ref_neg, med_pos, ref_pos, a))
  return(sinh_t(asi_corrected))
}

# ── Step 1: Reference peaks from B1 W48 ──────────────────────────
cat("Step 1: Computing reference peaks from B1 W48...\n")
b1_w48 <- list.files(file.path(FCS_DIR, "B1"), pattern = "W48.*[.]fcs$", full.names = TRUE)
w48_pool <- list()
for (f in b1_w48) {
  ff <- read.FCS(f, transformation = FALSE, truncate_max_range = FALSE)
  e <- exprs(ff)
  idx <- if (nrow(e) > 5000) sample(nrow(e), 5000) else 1:nrow(e)
  w48_pool[[length(w48_pool) + 1]] <- e[idx, , drop = FALSE]
  rm(ff, e); gc(verbose = FALSE)
}
w48_mat <- do.call(rbind, w48_pool)
cat(sprintf("  Pooled %d W48 cells from %d files\n", nrow(w48_mat), length(b1_w48)))

# CD45RA
cd45ra_asi <- asinh_t(w48_mat[, channels["CD45RA"]])
h <- hist(cd45ra_asi, breaks = 200, plot = FALSE)
vr <- which(h$mids > 0.1 & h$mids < 1.5)
v <- h$mids[vr[which.min(h$counts[vr])]]
ref_cd45ra_neg <- median(cd45ra_asi[cd45ra_asi <= v])
ref_cd45ra_pos <- median(cd45ra_asi[cd45ra_asi > v])
cat(sprintf("  CD45RA ref: neg=%.4f, pos=%.4f\n", ref_cd45ra_neg, ref_cd45ra_pos))

# CCR7
ccr7_asi <- asinh_t(w48_mat[, channels["CCR7"]])
h <- hist(ccr7_asi, breaks = 200, plot = FALSE)
vr <- which(h$mids > 0.0 & h$mids < 1.5)
v <- h$mids[vr[which.min(h$counts[vr])]]
ref_ccr7_neg <- median(ccr7_asi[ccr7_asi <= v])
ref_ccr7_pos <- median(ccr7_asi[ccr7_asi > v])
cat(sprintf("  CCR7   ref: neg=%.4f, pos=%.4f\n", ref_ccr7_neg, ref_ccr7_pos))

rm(w48_pool, w48_mat); gc(verbose = FALSE)

# ── Step 2: Compute shifts ───────────────────────────────────────
cat("\nStep 2: Computing shifts from bridge samples...\n")

get_medians <- function(path) {
  ff <- read.FCS(path, transformation = FALSE, truncate_max_range = FALSE)
  e <- exprs(ff)
  meds <- sapply(channels[shift_markers], function(ch) median(e[, ch]))
  names(meds) <- shift_markers
  rm(ff, e); gc(verbose = FALSE)
  meds
}

# ── 2a: B2→B1 shift from P27 + P33 ──
cat("\n  --- B2->B1 shift (P27 + P33 bridges) ---\n")

b1_p27 <- get_medians(file.path(FCS_DIR, "B1", "export_27 Basal_CD3 subset.fcs"))
b2_p27 <- get_medians(file.path(FCS_DIR, "B2", "export_27prep_CD3 subset.fcs"))
shift_b2_p27 <- b1_p27 - b2_p27

b1_p33 <- get_medians(file.path(FCS_DIR, "B1", "export_33 Basal_CD3 subset.fcs"))
b2_p33 <- get_medians(file.path(FCS_DIR, "B2", "export_33prep_CD3 subset.fcs"))
shift_b2_p33 <- b1_p33 - b2_p33

# Per-marker: if P27 and P33 have opposite signs, use P33 only
shift_b2 <- numeric(length(shift_markers))
names(shift_b2) <- shift_markers
for (m in shift_markers) {
  if (sign(shift_b2_p27[m]) != sign(shift_b2_p33[m]) &&
      (abs(shift_b2_p27[m]) > 50 || abs(shift_b2_p33[m]) > 50)) {
    shift_b2[m] <- shift_b2_p33[m]  # P33 only
  } else {
    shift_b2[m] <- (shift_b2_p27[m] + shift_b2_p33[m]) / 2
  }
}

cat("  P27 (B1-B2):  ")
for (m in shift_markers) cat(sprintf(" %+7.0f", shift_b2_p27[m]))
cat("\n  P33 (B1-B2):  ")
for (m in shift_markers) cat(sprintf(" %+7.0f", shift_b2_p33[m]))
cat("\n  Used:         ")
for (m in shift_markers) {
  src <- if (sign(shift_b2_p27[m]) != sign(shift_b2_p33[m]) &&
             (abs(shift_b2_p27[m]) > 50 || abs(shift_b2_p33[m]) > 50)) "P33" else "avg"
  cat(sprintf(" %7s", src))
}
cat("\n  Final:        ")
for (m in shift_markers) cat(sprintf(" %+7.0f", shift_b2[m]))
cat("\n")

# ── 2b: B3→B2 shift from P33 + P44 ──
cat("\n  --- B3->B2 shift (P33 + P44 bridges) ---\n")

b3_p33 <- get_medians(file.path(FCS_DIR, "B3", "export_33_CD3 subset.fcs"))
shift_b3b2_p33 <- b2_p33 - b3_p33

b2_p44 <- get_medians(file.path(FCS_DIR, "B3", "export_44HC_CD3 subset.fcs"))  # 44HC is B2
b3_p44 <- get_medians(file.path(FCS_DIR, "B3", "export_44_CD3 subset.fcs"))
shift_b3b2_p44 <- b2_p44 - b3_p44

# Per-marker: if P33 and P44 have opposite signs, use P33 only
shift_b3b2 <- numeric(length(shift_markers))
names(shift_b3b2) <- shift_markers
for (m in shift_markers) {
  if (sign(shift_b3b2_p33[m]) != sign(shift_b3b2_p44[m]) &&
      (abs(shift_b3b2_p33[m]) > 50 || abs(shift_b3b2_p44[m]) > 50)) {
    shift_b3b2[m] <- shift_b3b2_p33[m]  # P33 only
  } else {
    shift_b3b2[m] <- (shift_b3b2_p33[m] + shift_b3b2_p44[m]) / 2
  }
}

cat("  P33 (B2-B3):  ")
for (m in shift_markers) cat(sprintf(" %+7.0f", shift_b3b2_p33[m]))
cat("\n  P44 (B2-B3):  ")
for (m in shift_markers) cat(sprintf(" %+7.0f", shift_b3b2_p44[m]))
cat("\n  Used:         ")
for (m in shift_markers) {
  src <- if (sign(shift_b3b2_p33[m]) != sign(shift_b3b2_p44[m]) &&
             (abs(shift_b3b2_p33[m]) > 50 || abs(shift_b3b2_p44[m]) > 50)) "P33" else "avg"
  cat(sprintf(" %7s", src))
}
cat("\n  Final:        ")
for (m in shift_markers) cat(sprintf(" %+7.0f", shift_b3b2[m]))
cat("\n")

# ── 2c: B3→B1 total = B3→B2 + B2→B1 ──
shift_b3 <- shift_b3b2 + shift_b2

cat("\n  --- B3->B1 total (B3->B2 + B2->B1) ---\n")
cat("                ")
for (m in shift_markers) cat(sprintf(" %7s", m))
cat("\n  B2->B1:       ")
for (m in shift_markers) cat(sprintf(" %+7.0f", shift_b2[m]))
cat("\n  B3->B2:       ")
for (m in shift_markers) cat(sprintf(" %+7.0f", shift_b3b2[m]))
cat("\n  B3->B1 total: ")
for (m in shift_markers) cat(sprintf(" %+7.0f", shift_b3[m]))
cat("\n")

# ── Step 3: Process files ─────────────────────────────────────────
cat("\n\nStep 3: Processing files...\n\n")

# B1: reference, direct file copy (preserves original FCS format exactly)
cat("--- B1 (reference, file copy) ---\n")
b1_files <- list.files(file.path(FCS_DIR, "B1"), pattern = "[.]fcs$", full.names = TRUE)
for (f in b1_files) {
  bn <- basename(f)
  outpath <- file.path(outdir, "B1", paste0("norm_", bn))
  file.copy(f, outpath, overwrite = TRUE)
  cat(sprintf("  [B1-COPY] %s\n", bn))
}

# B2: peak alignment + median shift -> B1
# Includes files from /Library/Claude/CD3/B2/ AND 44HC from B3 folder
cat("\n--- B2 (peak alignment + median shift -> B1) ---\n")
b2_files <- list.files(file.path(FCS_DIR, "B2"), pattern = "[.]fcs$", full.names = TRUE)
b2_files <- c(b2_files, file.path(FCS_DIR, "B3", "export_44HC_CD3 subset.fcs"))

for (f in b2_files) {
  bn <- basename(f)
  ff <- read.FCS(f, transformation = FALSE, truncate_max_range = FALSE)
  ff <- fix_fcs_metadata(ff)
  e <- exprs(ff)
  cat(sprintf("  [B2->B1] %s (%d)\n", bn, nrow(e)))

  # 1) CD45RA/CCR7 peak alignment
  e[, channels["CD45RA"]] <- align_bimodal(
    e[, channels["CD45RA"]], ref_cd45ra_neg, ref_cd45ra_pos,
    valley_lo = 0.1, valley_hi = 1.5, marker_name = "CD45RA")
  e[, channels["CCR7"]] <- align_bimodal(
    e[, channels["CCR7"]], ref_ccr7_neg, ref_ccr7_pos,
    valley_lo = 0.0, valley_hi = 1.5, marker_name = "CCR7")

  # 2) Median shift for functional markers (B2 -> B1)
  for (m in shift_markers) {
    e[, channels[m]] <- e[, channels[m]] + shift_b2[m]
  }

  exprs(ff) <- e
  write.FCS(ff, file.path(outdir, "B2", paste0("norm_", bn)))
  rm(ff, e); gc(verbose = FALSE)
}

# B3: peak alignment + median shift -> B1 (via B2)
# EXCLUDE 44HC (it's B2)
cat("\n--- B3 (peak alignment + median shift -> B1) ---\n")
b3_files <- list.files(file.path(FCS_DIR, "B3"), pattern = "[.]fcs$", full.names = TRUE)
b3_files <- b3_files[!grepl("44HC", b3_files)]

for (f in b3_files) {
  bn <- basename(f)
  ff <- read.FCS(f, transformation = FALSE, truncate_max_range = FALSE)
  ff <- fix_fcs_metadata(ff)
  e <- exprs(ff)
  cat(sprintf("  [B3->B1] %s (%d)\n", bn, nrow(e)))

  # 1) CD45RA/CCR7 peak alignment
  e[, channels["CD45RA"]] <- align_bimodal(
    e[, channels["CD45RA"]], ref_cd45ra_neg, ref_cd45ra_pos,
    valley_lo = 0.1, valley_hi = 1.5, marker_name = "CD45RA")
  e[, channels["CCR7"]] <- align_bimodal(
    e[, channels["CCR7"]], ref_ccr7_neg, ref_ccr7_pos,
    valley_lo = 0.0, valley_hi = 1.5, marker_name = "CCR7")

  # 2) Median shift for functional markers (B3 -> B1 total)
  for (m in shift_markers) {
    e[, channels[m]] <- e[, channels[m]] + shift_b3[m]
  }

  exprs(ff) <- e
  write.FCS(ff, file.path(outdir, "B3", paste0("norm_", bn)))
  rm(ff, e); gc(verbose = FALSE)
}

# ── Step 4: Verification ─────────────────────────────────────────
cat("\n\n============================================================\n")
cat("  Verification: Bridge medians (arcsinh)\n")
cat("============================================================\n\n")

all_markers <- c("CD45RA", "CCR7", shift_markers)

get_all_meds <- function(path) {
  ff <- read.FCS(path, transformation = FALSE, truncate_max_range = FALSE)
  e <- exprs(ff)
  meds <- sapply(channels[all_markers], function(ch) median(asinh_t(e[, ch])))
  names(meds) <- all_markers
  rm(ff, e); gc(verbose = FALSE)
  meds
}

# P33: B1 vs B2 vs B3
cat("--- P33 (B1 / B2 / B3) ---\n")
cat(sprintf("%-10s", ""))
for (m in all_markers) cat(sprintf(" %7s", m))
cat("\n")

b1_m <- get_all_meds(file.path(FCS_DIR, "B1", "export_33 Basal_CD3 subset.fcs"))
b2_orig <- get_all_meds(file.path(FCS_DIR, "B2", "export_33prep_CD3 subset.fcs"))
b2_norm <- get_all_meds(file.path(outdir, "B2/norm_export_33prep_CD3 subset.fcs"))
b3_orig <- get_all_meds(file.path(FCS_DIR, "B3", "export_33_CD3 subset.fcs"))
b3_norm <- get_all_meds(file.path(outdir, "B3/norm_export_33_CD3 subset.fcs"))

cat(sprintf("%-10s", "B1"))
for (m in all_markers) cat(sprintf(" %+7.3f", b1_m[m]))
cat("\n")
cat(sprintf("%-10s", "B2 bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_orig[m]))
cat("\n")
cat(sprintf("%-10s", "B2 aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_norm[m]))
cat("\n")
cat(sprintf("%-10s", "B3 bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_orig[m]))
cat("\n")
cat(sprintf("%-10s", "B3 aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_norm[m]))
cat("\n\n")

cat("Gaps vs B1:\n")
cat(sprintf("%-10s", "B2 bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_orig[m] - b1_m[m]))
cat("\n")
cat(sprintf("%-10s", "B2 aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_norm[m] - b1_m[m]))
cat("\n")
cat(sprintf("%-10s", "B3 bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_orig[m] - b1_m[m]))
cat("\n")
cat(sprintf("%-10s", "B3 aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_norm[m] - b1_m[m]))
cat("\n\n")

# P27: B1 vs B2
cat("--- P27 (B1 / B2) ---\n")
cat(sprintf("%-10s", ""))
for (m in all_markers) cat(sprintf(" %7s", m))
cat("\n")
b1_27 <- get_all_meds(file.path(FCS_DIR, "B1", "export_27 Basal_CD3 subset.fcs"))
b2_27o <- get_all_meds(file.path(FCS_DIR, "B2", "export_27prep_CD3 subset.fcs"))
b2_27n <- get_all_meds(file.path(outdir, "B2/norm_export_27prep_CD3 subset.fcs"))
cat(sprintf("%-10s", "B1"))
for (m in all_markers) cat(sprintf(" %+7.3f", b1_27[m]))
cat("\n")
cat(sprintf("%-10s", "B2 bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_27o[m]))
cat("\n")
cat(sprintf("%-10s", "B2 aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_27n[m]))
cat("\n")
cat(sprintf("%-10s", "gap bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_27o[m] - b1_27[m]))
cat("\n")
cat(sprintf("%-10s", "gap aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_27n[m] - b1_27[m]))
cat("\n\n")

# P44: B2(44HC) vs B3(44)
cat("--- P44 (B2[44HC] / B3[44]) ---\n")
cat(sprintf("%-10s", ""))
for (m in all_markers) cat(sprintf(" %7s", m))
cat("\n")
b2_44o <- get_all_meds(file.path(FCS_DIR, "B3", "export_44HC_CD3 subset.fcs"))
b2_44n <- get_all_meds(file.path(outdir, "B2/norm_export_44HC_CD3 subset.fcs"))
b3_44o <- get_all_meds(file.path(FCS_DIR, "B3", "export_44_CD3 subset.fcs"))
b3_44n <- get_all_meds(file.path(outdir, "B3/norm_export_44_CD3 subset.fcs"))
cat(sprintf("%-10s", "B2 bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_44o[m]))
cat("\n")
cat(sprintf("%-10s", "B2 aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_44n[m]))
cat("\n")
cat(sprintf("%-10s", "B3 bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_44o[m]))
cat("\n")
cat(sprintf("%-10s", "B3 aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_44n[m]))
cat("\n")
cat(sprintf("%-10s", "gap bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_44o[m] - b2_44o[m]))
cat("\n")
cat(sprintf("%-10s", "gap aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_44n[m] - b2_44n[m]))
cat("\n\n")

# P47: B2 vs B3
cat("--- P47 (B2 / B3) ---\n")
cat(sprintf("%-10s", ""))
for (m in all_markers) cat(sprintf(" %7s", m))
cat("\n")
b2_47o <- get_all_meds(file.path(FCS_DIR, "B2", "export_47HC_CD3 subset.fcs"))
b2_47n <- get_all_meds(file.path(outdir, "B2/norm_export_47HC_CD3 subset.fcs"))
b3_47o <- get_all_meds(file.path(FCS_DIR, "B3", "export_47_CD3 subset.fcs"))
b3_47n <- get_all_meds(file.path(outdir, "B3/norm_export_47_CD3 subset.fcs"))
cat(sprintf("%-10s", "B2 bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_47o[m]))
cat("\n")
cat(sprintf("%-10s", "B2 aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_47n[m]))
cat("\n")
cat(sprintf("%-10s", "B3 bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_47o[m]))
cat("\n")
cat(sprintf("%-10s", "B3 aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_47n[m]))
cat("\n")
cat(sprintf("%-10s", "gap bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_47o[m] - b2_47o[m]))
cat("\n")
cat(sprintf("%-10s", "gap aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_47n[m] - b2_47n[m]))
cat("\n\n")

# P43: B2 vs B3
cat("--- P43 (B2 / B3) ---\n")
cat(sprintf("%-10s", ""))
for (m in all_markers) cat(sprintf(" %7s", m))
cat("\n")
b2_43o <- get_all_meds(file.path(FCS_DIR, "B2", "export_43HC_CD3 subset.fcs"))
b2_43n <- get_all_meds(file.path(outdir, "B2/norm_export_43HC_CD3 subset.fcs"))
b3_43o <- get_all_meds(file.path(FCS_DIR, "B3", "export_43_CD3 subset.fcs"))
b3_43n <- get_all_meds(file.path(outdir, "B3/norm_export_43_CD3 subset.fcs"))
cat(sprintf("%-10s", "B2 bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_43o[m]))
cat("\n")
cat(sprintf("%-10s", "B2 aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b2_43n[m]))
cat("\n")
cat(sprintf("%-10s", "B3 bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_43o[m]))
cat("\n")
cat(sprintf("%-10s", "B3 aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_43n[m]))
cat("\n")
cat(sprintf("%-10s", "gap bef"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_43o[m] - b2_43o[m]))
cat("\n")
cat(sprintf("%-10s", "gap aft"))
for (m in all_markers) cat(sprintf(" %+7.3f", b3_43n[m] - b2_43n[m]))
cat("\n\n")

cat("============================================================\n")
cat(sprintf("B1: %d files, B2: %d files, B3: %d files\n",
    length(list.files(file.path(outdir, "B1"), pattern = "[.]fcs$")),
    length(list.files(file.path(outdir, "B2"), pattern = "[.]fcs$")),
    length(list.files(file.path(outdir, "B3"), pattern = "[.]fcs$"))))
cat(sprintf("Output: %s\n", outdir))
cat("=== V11 Done! ===\n")
gc()
