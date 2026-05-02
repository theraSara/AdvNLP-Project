library(lme4)
library(lmerTest)

run_benchmark <- function(input_csv, predictor_col, predictor_label, output_dir, subset_mode = "all") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  df <- read.csv(input_csv)

  df$SUB <- as.factor(df$SUB)
  df$ITEM <- as.factor(df$ITEM)
  df$condition <- as.factor(df$condition)

  if (!("swp_target_present" %in% names(df)) && ("swp_target_rank" %in% names(df))) {
    df$swp_target_present <- !is.na(df$swp_target_rank)
  }

  required_cols <- c(
    "SUB", "ITEM", "condition", "log_rt",
    "word_length", "word_frequency",
    "cloze_surprisal", predictor_col
  )

  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
  }

  df_model <- df[!is.na(df[[predictor_col]]), ]

  if (subset_mode == "swp_present_only") {
    if (!("swp_target_present" %in% names(df_model))) {
      stop("subset_mode = swp_present_only requested, but swp_target_present could not be derived.")
    }
    df_model <- df_model[df_model$swp_target_present == TRUE, ]
  }

  n_rows <- nrow(df_model)
  n_items <- length(unique(df_model$ITEM))

  cat("\n=============================\n")
  cat("Running:", predictor_label, "\n")
  cat("Predictor column:", predictor_col, "\n")
  cat("Subset mode:", subset_mode, "\n")
  cat("Rows used:", n_rows, "\n")
  cat("Unique items:", n_items, "\n")
  cat("Output dir:", output_dir, "\n")
  cat("=============================\n")

  base_formula <- as.formula(
    "log_rt ~ word_length + word_frequency + (1 | SUB) + (1 | ITEM)"
  )

  cloze_formula <- as.formula(
    "log_rt ~ cloze_surprisal + word_length + word_frequency + (1 | SUB) + (1 | ITEM)"
  )

  pred_formula <- as.formula(
    paste0("log_rt ~ ", predictor_col, " + word_length + word_frequency + (1 | SUB) + (1 | ITEM)")
  )

  both_formula <- as.formula(
    paste0("log_rt ~ cloze_surprisal + ", predictor_col, " + word_length + word_frequency + (1 | SUB) + (1 | ITEM)")
  )

  m_base <- lmer(base_formula, data = df_model, REML = FALSE)
  m_cloze <- lmer(cloze_formula, data = df_model, REML = FALSE)
  m_pred  <- lmer(pred_formula,  data = df_model, REML = FALSE)
  m_both  <- lmer(both_formula,  data = df_model, REML = FALSE)

  models <- list(
    m_base = m_base,
    m_cloze = m_cloze,
    m_pred = m_pred,
    m_both = m_both
  )

  extract_fixed_effects <- function(model, model_name, predictor_label, subset_mode, n_rows, n_items) {
    s <- summary(model)
    coefs <- as.data.frame(s$coefficients)
    coefs$term <- rownames(coefs)
    rownames(coefs) <- NULL

    names(coefs) <- make.names(names(coefs))
    coefs$model <- model_name
    coefs$predictor_label <- predictor_label
    coefs$subset_mode <- subset_mode
    coefs$n_rows <- n_rows
    coefs$n_items <- n_items

    coefs <- coefs[, c("predictor_label", "subset_mode", "n_rows", "n_items", "model", "term", "Estimate", "Std..Error", "df", "t.value", "Pr...t..")]
    names(coefs) <- c("predictor_label", "subset_mode", "n_rows", "n_items", "model", "term", "estimate", "std_error", "df", "t_value", "p_value")
    coefs
  }

  coef_table <- do.call(
    rbind,
    lapply(names(models), function(nm) {
      extract_fixed_effects(models[[nm]], nm, predictor_label, subset_mode, n_rows, n_items)
    })
  )

  write.csv(coef_table, file.path(output_dir, "coefficients.csv"), row.names = FALSE)

  aic_table <- do.call(
    rbind,
    lapply(names(models), function(nm) {
      m <- models[[nm]]
      data.frame(
        predictor_label = predictor_label,
        subset_mode = subset_mode,
        n_rows = n_rows,
        n_items = n_items,
        model = nm,
        npar = attr(logLik(m), "df"),
        AIC = AIC(m),
        BIC = BIC(m),
        logLik = as.numeric(logLik(m))
      )
    })
  )

  write.csv(aic_table, file.path(output_dir, "aic.csv"), row.names = FALSE)

  extract_anova_row <- function(m1, m2, name1, name2, comparison_name, predictor_label, subset_mode, n_rows, n_items) {
    a <- as.data.frame(anova(m1, m2))

    data.frame(
      predictor_label = predictor_label,
      subset_mode = subset_mode,
      n_rows = n_rows,
      n_items = n_items,
      comparison = comparison_name,
      model_1 = name1,
      model_2 = name2,
      npar_1 = a$npar[1],
      npar_2 = a$npar[2],
      AIC_1 = a$AIC[1],
      AIC_2 = a$AIC[2],
      BIC_1 = a$BIC[1],
      BIC_2 = a$BIC[2],
      logLik_1 = a$logLik[1],
      logLik_2 = a$logLik[2],
      Chisq = a$Chisq[2],
      Df = a$Df[2],
      p_value = a$`Pr(>Chisq)`[2]
    )
  }

  anova_table <- rbind(
    extract_anova_row(m_base,  m_cloze, "m_base",  "m_cloze", "cloze_vs_base", predictor_label, subset_mode, n_rows, n_items),
    extract_anova_row(m_base,  m_pred,  "m_base",  "m_pred",  "predictor_vs_base", predictor_label, subset_mode, n_rows, n_items),
    extract_anova_row(m_cloze, m_both,  "m_cloze", "m_both",  "predictor_beyond_cloze", predictor_label, subset_mode, n_rows, n_items),
    extract_anova_row(m_pred,  m_both,  "m_pred",  "m_both",  "cloze_beyond_predictor", predictor_label, subset_mode, n_rows, n_items)
  )

  write.csv(anova_table, file.path(output_dir, "anova.csv"), row.names = FALSE)

  sink(file.path(output_dir, "model_summaries.txt"))
  cat("Predictor:", predictor_col, "\n")
  cat("Predictor label:", predictor_label, "\n")
  cat("Subset mode:", subset_mode, "\n")
  cat("Rows used:", n_rows, "\n")
  cat("Unique items:", n_items, "\n\n")

  cat("=== m_base ===\n")
  print(summary(m_base))
  cat("\n=== m_cloze ===\n")
  print(summary(m_cloze))
  cat("\n=== m_pred ===\n")
  print(summary(m_pred))
  cat("\n=== m_both ===\n")
  print(summary(m_both))
  sink()

  invisible(list(
    coefficients = coef_table,
    aic = aic_table,
    anova = anova_table
  ))
}