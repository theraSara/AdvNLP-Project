source("LME/benchmark_utils.R")

run_coverage_controlled <- function() {
  input_csv <- "data/master_modeling_data_with_swp.csv"
  dataset_label <- "swp_present_only"

  df_check <- read.csv(input_csv, stringsAsFactors = FALSE)

  uni_specs <- list(
    list(col = "gpt2_uni_surprisal",      label = "gpt2_uni"),
    list(col = "gemma270m_uni_surprisal", label = "gemma270m_uni"),
    list(col = "gemma12b_uni_surprisal",  label = "gemma12b_uni"),
    list(col = "swp_surprisal_bits",      label = "swp")
  )

  run_if_present <- function(spec) {
    if (!(spec$col %in% names(df_check))) {
      cat("Skipping missing column:", spec$col, "\n")
      return(NULL)
    }

    if (all(is.na(df_check[[spec$col]]))) {
      cat("Skipping all-NA predictor:", spec$col, "\n")
      return(NULL)
    }

    run_benchmark(
      input_csv = input_csv,
      predictor_col = spec$col,
      predictor_label = spec$label,
      output_dir = file.path(
        "regression_output_coverage_controlled",
        dataset_label,
        "primary_uni",
        spec$label
      ),
      subset_mode = "swp_present_only"
    )
  }

  for (spec in uni_specs) {
    run_if_present(spec)
  }

  cat("\nFinished coverage-controlled benchmark.\n")
}

run_coverage_controlled()