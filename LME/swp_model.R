source("LME/benchmark_utils.R")

run_benchmark(
  input_csv = "data/master_modeling_data_with_swp.csv",
  predictor_col = "swp_surprisal_bits",
  predictor_label = "swp", # swp
  output_dir = file.path("regression_output", "primary_uni", "swp"),
  subset_mode = "swp_present_only" # all
)