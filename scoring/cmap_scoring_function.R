#! /usr/bin/env Rscript

################################################
# Score Inference Algorithm Submissions        #
# The Connectivity Map at The Broad Institute  #
# with HBS and Topcoder                        #
# Spring 2016                                  #
################################################

if (!(require('argparser'))) {
  message('argparser library not found, installing...')
  chooseCRANmirror(81)
  install.packages('argparser')
}

library(argparser)

# create arg parser for parsing command line arguments
parser <- arg_parser("
 Score inference algorithm results. Computes the Relative
 Inference Score (RIS) for a matrix of inferred gene expression
 values, and the average RIS across all genes. Outputs a summary table
 with the gene-level RIS values and a file, RIS.txt, with the
 mean RIS across all genes.

 example:
 cmap_scoring_funtion.R --inf_ds /path/to/inference/result.csv \
    --truth_ds /path/to/truth/matrix.csv \
    --reference_scores /path/to/reference/scores.csv \
    --out /desired/output/path
                     ")
parser <- add_argument(parser, "--inf_ds",
                       help="Path to matrix with inferred values. Expected to be a
                       headerless CSV with one row per gene and one column per sample.
                       Row and column order is assumed to be in sync with --truth_ds.",
                       default=NA)
parser <- add_argument(parser, "--truth_ds",
                      help="Path to matrix with ground truth values. Expected to be a
                       headerless CSV with one row per gene and one column per sample.
                       Row and column order is assumed to be in sync with --inf_ds.",
                       default=NA)
parser <- add_argument(parser, "--reference_scores",
                       help="Path to 1-column CSV containing a vector of gene-level
                         scores derived on the existing CMap inference model.",
                       default=NA)
parser <- add_argument(parser, "--gene_dim",
                       help="Indicates which dimension in --inf_ds and --truth_ds 
                         corresponds to genes. Assumed to be the same for both matrices.
                         Must be one of ['row', 'column'].",
                       default="row")
parser <- add_argument(parser, "--out", help="output path", default=getwd())

args <- parse_args(parser, argv=commandArgs(trailingOnly=T))


### Function Definitions ###

# fast correlations
fast_cor <- function(X, Y) {
  # convert to ranks by column
  X <- apply(X, 2, rank, ties.method = "average")
  Y <- apply(Y, 2, rank, ties.method = "average")
  # zscore columns
  X <- scale(X)
  Y <- scale(Y)
  # spearman
  nsample <- nrow(X)
  corr <- (t(X) %*% Y) / (nsample - 1)
  return(corr)
}

# THE RELATIVE SCORING FUNCTION
# computes the relative imputation score (RIS)
get_relative_score <- function(ref_scores, test_scores, coefficient=1e6) {
  numerator <- 2 - ref_scores
  denominator <- 2 - test_scores
  scores <- numerator / denominator
  scores <- coefficient * scores
  return(scores)
}

# rescale the relative score
# assumes current_score is the aggregated RIS
get_rescaled_score <- function(current_score, score_ceiling, par_score=1e6) {
  # ScoreScaled = CurrentScore (ScoringCeiling - 1M)/(ScoringCeiling - CurrentScore)
  score_scaled <- current_score * ((score_ceiling - par_score) / (score_ceiling - current_score))
  return(score_scaled)
}


### Main Program ###

# make sure required args are supplied
req_args <- c("inf_ds", "truth_ds", "reference_scores", "gene_dim")
missing_args <- setdiff(req_args, names(args))
if (length(missing_args) > 0) {
  stop(paste("the following required arguments are missing:\n",
             paste(missing_args, collapse="\n")))
}

# make sure the gene_dim argument is valid
if (!args$gene_dim %in% c("row", "column")) {
  stop("gene_dim must be either 'row' or 'column'")
}

# set up output path
outpath <- args$out
if (!file.exists(outpath)) dir.create(outpath)

# read the user's submission
# expected to be a CSV containing
# the predicted expression levels
inf <- as.matrix(read.csv(args$inf_ds, header=F))

# read the ground truth data
# expected to be a CSV containing
# the ground truth expression levels
truth <- as.matrix(read.csv(args$truth_ds, header=F))

# make sure matrices are the same dimensions
if (nrow(inf) != nrow(truth) | ncol(inf) != ncol(truth)) {
  stop("matrices are of different dimensions")
}

# read the baseline performance table
# expect this to be a single-column table
base_performance <- as.vector(read.csv(args$reference_scores, header=F)[, 1])

# need to get genes on the columns, so tranpose
# matrices if necessary
if (args$gene_dim == "row") {
  message("transposing matrices so genes are along columns")
  inf <- t(inf)
  truth <- t(truth)
}

# compute correlations
message("computing correlations...")
corr <- fast_cor(inf, truth)
message("done")

# rank across columns (inf genes on rows)
# asks how well the inferred gene pulls up its
# measured counterpart
# higher values get higher rank
rank_by_row <- apply(corr, 1, function(x) rank(x))

# convert to fraction ranks
frac_ranks <- rank_by_row / nrow(corr)

# collate the absolute correlation and the
# fraction rank into a single table
df <- data.frame(correlation = diag(corr),
                 fraction_rank = diag(frac_ranks))

# average the correlation and fraction rank to get the score
# threshold negative correlations to zero
df$thresholded_correlation <- sapply(df$correlation, function(x) max(x, 0))
df$score_test <- (df$thresholded_correlation + df$fraction_rank) / 2 

# and get the reference score relative to baseline
df$score_ref <- base_performance
df$RIS <- get_relative_score(df$score_ref, df$score_test)

# figure out the max possible score by supplying
# a vector of all perfect scores (all 1)
best_ris <- get_relative_score(df$score_ref, rep(1, nrow(df)))
mean_best_ris <- mean(best_ris)

# dump the summary table
write.csv(df, paste(outpath, "summary_table.csv", sep="/"),
            row.names=F, quote=F)

# compute some summary stats
mean_RIS <- mean(df$RIS, na.rm=T)

# rescaled mean_RIS
rescaled_mean_RIS <- get_rescaled_score(mean_RIS, mean_best_ris)

# dump a file with the mean RIS
write(paste("RIS:", round(rescaled_mean_RIS, 2)), file=paste(outpath, "RIS.txt", sep="/"),
      append = T)

# print a summary to the screen
message("---------- Summary ----------")
message(paste("Mean RIS:", round(rescaled_mean_RIS, 2)))