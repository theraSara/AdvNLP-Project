collect_csvs <- function(base_dir, filename) {
  files <- list.files(base_dir, pattern = filename, recursive = TRUE, full.names = TRUE)
  if (length(files) == 0) return(data.frame())

  dfs <- lapply(files, function(f) {
    df <- read.csv(f, stringsAsFactors = FALSE)
    df$source_file <- f
    df
  })

  do.call(rbind, dfs)
}

base_dir <- "regression_output_coverage_controlled/swp_present_only"

all_aic  <- collect_csvs(base_dir, "^aic\\.csv$")
all_anova <- collect_csvs(base_dir, "^anova\\.csv$")
all_coef <- collect_csvs(base_dir, "^coefficients\\.csv$")

dir.create(file.path(base_dir, "_collected"), showWarnings = FALSE, recursive = TRUE)

write.csv(all_aic,  file.path(base_dir, "_collected", "all_aic.csv"), row.names = FALSE)
write.csv(all_anova, file.path(base_dir, "_collected", "all_anova.csv"), row.names = FALSE)
write.csv(all_coef, file.path(base_dir, "_collected", "all_coefficients.csv"), row.names = FALSE)

cat("Saved collected files to:\n")
cat(file.path(base_dir, "_collected", "all_aic.csv"), "\n")
cat(file.path(base_dir, "_collected", "all_anova.csv"), "\n")
cat(file.path(base_dir, "_collected", "all_coefficients.csv"), "\n")