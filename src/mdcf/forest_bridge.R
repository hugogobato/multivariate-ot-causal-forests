args <- commandArgs(trailingOnly = TRUE)
action <- args[[1]]
method <- args[[2]]
root <- args[[3]]
ncol_x <- as.integer(args[[4]])
n_test <- as.integer(args[[5]])

if (method == "grf") {
  suppressPackageStartupMessages(library(grf))
} else if (method == "cdrf") {
  predecessor <- args[[6]]
  source(file.path(predecessor, "research", "baselines", "causal_drf_r", "causal_drf.R"))
} else {
  stop("unknown native forest")
}

read_values <- function(path, count) {
  conn <- file(path, "rb")
  on.exit(close(conn))
  values <- readBin(conn, what = "double", n = count, size = 8, endian = "little")
  if (length(values) != count) stop("truncated binary bridge matrix")
  values
}

for (rotation in 0:2) {
  data <- file.path(root, as.character(rotation))
  meta <- scan(file.path(data, "meta"), quiet = TRUE)
  n <- meta[[1]]
  outputs <- meta[[2]]
  num_trees <- as.integer(meta[[3]])
  min_arm <- as.integer(meta[[4]])
  max_depth <- as.integer(meta[[5]])
  fraction <- meta[[6]]
  seed <- as.integer(meta[[7]])
  bandwidth <- meta[[8]]
  model_path <- file.path(data, "model.rds")
  if (action == "fit") {
    X <- matrix(read_values(file.path(data, "X"), n * ncol_x), nrow = n, byrow = TRUE)
    W <- as.integer(read_values(file.path(data, "W"), n))
    Y <- matrix(read_values(file.path(data, "Y"), n * outputs), nrow = n, byrow = TRUE)
    if (method == "grf") {
      means <- matrix(read_values(file.path(data, "mu"), n * outputs), nrow = n, byrow = TRUE)
      e <- read_values(file.path(data, "e"), n)
      fit <- multi_arm_causal_forest(
        X, Y, factor(W, levels = c(0, 1)), Y.hat = means, W.hat = cbind(1 - e, e),
        num.trees = num_trees, min.node.size = 2L * min_arm,
        sample.fraction = fraction, honesty = TRUE, honesty.fraction = 0.5,
        alpha = 0.05, sample.weights = read_values(file.path(data, "weights"), n),
        num.threads = 1L, compute.oob.predictions = FALSE, seed = seed
      )
    } else {
      divisors <- which(num_trees %% seq_len(min(num_trees, 8L)) == 0L)
      fit <- causal_drf_fit(
        X, W, Y, num_trees = num_trees, num_groups = max(divisors),
        num_features = 128L, min_arm_leaf = min_arm, min_arm_build = min_arm,
        sample_fraction = fraction, honesty_fraction = 0.5,
        max_depth = max_depth, bandwidth = bandwidth, seed = seed
      )
    }
    saveRDS(fit, model_path, compress = FALSE)
  } else if (action == "predict") {
    fit <- readRDS(model_path)
    X_test <- matrix(read_values(file.path(root, "X_test"), n_test * ncol_x), nrow = n_test, byrow = TRUE)
    if (method == "grf") {
      weights <- as.matrix(get_forest_weights(fit, X_test, num.threads = 1L))
    } else {
      arms <- causal_drf_weights(fit, X_test)
      weights <- 0.5 * (arms$treated + arms$control)
    }
    writeBin(as.double(t(weights)), file.path(data, "kernel"), size = 8, endian = "little")
  } else {
    stop("unknown forest bridge action")
  }
}
