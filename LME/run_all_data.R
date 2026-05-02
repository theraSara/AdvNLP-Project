source("LME/benchmark_utils.R")

run_dataset_suite <- function(input_csv, dataset_label, include_gemma12b_raw = FALSE) {
  uni_specs <- list(
    list(col = "gpt2_uni_surprisal",      label = "gpt2_uni"),
    list(col = "gemma270m_uni_surprisal", label = "gemma270m_uni"),
    list(col = "gemma12b_uni_surprisal",  label = "gemma12b_uni"),
  )

  df_check <- read.csv(input_csv, stringsAsFactors = FALSE)

  run_if_present <- function(spec, group_name) {
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
      output_dir = file.path("regression_output", dataset_label, group_name, spec$label)
    )
  }

  for (spec in uni_specs) {
    run_if_present(spec, "primary_uni")
  }

  cat("\nFinished dataset:", dataset_label, "\n")
}

run_dataset_suite(
  input_csv = "data_output/master_modeling_data.csv",
  dataset_label = "filtered_17k"
)
