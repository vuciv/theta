#!/usr/bin/env Rscript
# Fit a 1PL/2PL model with mirt and report the *fit-only* wall time + item params.
#
#   Rscript bench/mirt_fit.R <data_csv> <model:1PL|2PL> <out_csv> [quadpts] [repeats]
#
# Data CSV is [persons x items], no header (same orientation theta uses). mirt is
# C++ (no per-call compile), but we still warm up once and take the min of a few
# timed fits so the number is comparable to theta-warm and girth. Item params are
# written to <out_csv> in the IRT (difficulty) parametrization; the fit time and
# convergence flag are printed to stdout as `TIME <sec>` / `CONVERGED <bool>`.

suppressMessages(library(mirt))

args <- commandArgs(trailingOnly = TRUE)
data_csv <- args[1]
model    <- args[2]
out_csv  <- args[3]
quadpts  <- if (length(args) >= 4) as.integer(args[4]) else 61L
repeats  <- if (length(args) >= 5) as.integer(args[5]) else 3L

dat <- as.matrix(read.csv(data_csv, header = FALSE))
J <- ncol(dat)

fit_once <- function() {
  if (model == "1PL") {
    # common (estimated) slope across items == theta's 1PL; equate all a1
    syntax <- sprintf("F = 1-%d\nCONSTRAIN = (1-%d, a1)", J, J)
    mirt(dat, mirt.model(syntax), itemtype = "2PL", method = "EM",
         quadpts = quadpts, verbose = FALSE,
         technical = list(NCYCLES = 5000))
  } else {
    mirt(dat, 1, itemtype = "2PL", method = "EM",
         quadpts = quadpts, verbose = FALSE,
         technical = list(NCYCLES = 5000))
  }
}

invisible(fit_once())  # warm up

best <- Inf
mod <- NULL
for (i in seq_len(repeats)) {
  el <- system.time({ m <- fit_once() })[["elapsed"]]
  if (el < best) { best <- el; mod <- m }
}

items <- coef(mod, simplify = TRUE, IRTpars = TRUE)$items
write.csv(data.frame(a = items[, "a"], b = items[, "b"]), out_csv, row.names = FALSE)

cat(sprintf("TIME %.6f\n", best))
cat(sprintf("CONVERGED %s\n", as.character(extract.mirt(mod, "converged"))))
