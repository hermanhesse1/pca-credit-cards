# ===== Optimized Credit Default Classification with Tidymodels =====

# 1. Load Libraries
library(tidymodels)   # Core modeling (recipes, parsnip, tune, workflows, yardstick, etc.)
library(themis)       # Class‐imbalance methods
library(vip)          # Variable importance
library(doFuture)     # Parallel backend
library(finetune)     # Efficient tuning (race_anova)
library(lime)         # Local explanations
library(probably)     # Threshold utilities
library(tictoc)       # Time tracking
# caret::findCorrelation is called via caret:: so no need to library(caret)

# 2. Parallel Processing Setup
registerDoFuture()
workers <- max(1, parallel::detectCores() - 1)
plan(multisession, workers = workers)
cat("Using", workers, "parallel workers\n")
tic("Total Execution Time")

# 3. Load and Clean Data
cat("Loading and cleaning data...\n")
data_raw <- readr::read_csv("ohe28_cleaned_data.csv", show_col_types = FALSE) %>%
  janitor::clean_names() %>%
  mutate(
    default = factor(default, levels = c(0, 1),
                     labels = c("no_default", "default"))
  )

# Quick checks
glimpse(data_raw)
cat("Missing values present?", anyNA(data_raw), "\n")
cat("Class distribution:\n")
print(table(data_raw$default))
cat("Default rate:", scales::percent(mean(data_raw$default == "default")), "\n")

# 4. Train/Test Split
cat("Creating train/test split...\n")
set.seed(2025)
split_obj  <- initial_split(data_raw, prop = 0.8, strata = default)
train_data <- training(split_obj)
test_data  <- testing(split_obj)

# 5. Exploratory Correlation Check
cat("Checking feature correlations...\n")
numeric_vars <- train_data %>% select(where(is.numeric))
if (ncol(numeric_vars) > 1) {
  cor_mat   <- cor(numeric_vars, use = "pairwise.complete.obs")
  high_corr <- caret::findCorrelation(cor_mat, cutoff = 0.8, names = TRUE)
  if (length(high_corr) > 0) {
    cat("Highly correlated features detected:", toString(high_corr), "\n")
  } else {
    cat("No highly correlated features found.\n")
  }
}

# 6. Define preprocessing recipes
cat("Creating preprocessing recipes...\n")
base_recipe <- recipe(default ~ ., data = train_data) %>%
  step_zv(all_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.8) %>%
  step_normalize(all_numeric_predictors())

recipe_downsample <- base_recipe %>%
  step_downsample(default, under_ratio = tune()) %>%
  step_pca(all_numeric_predictors(), threshold = tune())

recipe_no_balance <- base_recipe %>%
  step_pca(all_numeric_predictors(), threshold = tune())

# 7. Define models
knn_spec <- nearest_neighbor(
  neighbors   = tune(),
  weight_func = tune(),
  dist_power  = tune()
) %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

rf_specs <- list(
  RF_w2 = rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
    set_engine("ranger", importance = "permutation",
               class.weights = c(no_default = 1, default = 2)) %>%
    set_mode("classification"),
  RF_w3 = rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
    set_engine("ranger", importance = "permutation",
               class.weights = c(no_default = 1, default = 3)) %>%
    set_mode("classification"),
  RF_w4 = rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
    set_engine("ranger", importance = "permutation",
               class.weights = c(no_default = 1, default = 4)) %>%
    set_mode("classification")
)

# 8. Set up workflows
cat("Setting up workflows...\n")
wf_knn_downsample <- workflow() %>%
  add_model(knn_spec) %>%
  add_recipe(recipe_downsample)

rf_workflows <- purrr::map(rf_specs, function(spec) {
  workflow() %>%
    add_model(spec) %>%
    add_recipe(recipe_no_balance)
})
names(rf_workflows) <- names(rf_specs)

# 9. Resampling setup
cat("Creating resampling folds...\n")
set.seed(2025)
folds <- vfold_cv(train_data, v = 5, strata = default)

# 10. Hyperparameter tuning grids
cat("Setting up tuning grids...\n")
knn_grid <- grid_latin_hypercube(
  neighbors(range = c(3, 20)),
  weight_func(values = c("rectangular", "triangular", "epanechnikov", "gaussian")),
  dist_power(range = c(1, 2)),
  threshold(range = c(0.7, 0.95)),     # PCA variance
  under_ratio(range = c(1.5, 3)),      # Downsampling ratio
  size = 15
)
rf_grid <- grid_latin_hypercube(
  mtry(range = c(5, 20)),
  min_n(range = c(2, 10)),
  threshold(range = c(0.7, 0.95)),     # PCA variance
  size = 15
)

# 11. Race control & metrics
ctrl <- control_race(
  save_pred     = TRUE,
  parallel_over = "resamples",
  allow_par     = TRUE,
  save_workflow = TRUE,
  verbose       = TRUE
)
model_metrics <- metric_set(roc_auc, accuracy, pr_auc, sens, spec)

# 12. Tune
cat("Tuning models...\n")
tic("KNN tuning")
tune_knn_down <- tune_race_anova(
  wf_knn_downsample,
  resamples = folds,
  grid      = knn_grid,
  metrics   = model_metrics,
  control   = ctrl
)
toc()

rf_tunes <- list()
for (nm in names(rf_workflows)) {
  tic(paste("RF tuning:", nm))
  rf_tunes[[nm]] <- tune_race_anova(
    rf_workflows[[nm]],
    resamples = folds,
    grid      = rf_grid,
    metrics   = model_metrics,
    control   = ctrl
  )
  toc()
}

# 13. Compare model performance
cat("Comparing model performance...\n")
knn_best <- collect_metrics(tune_knn_down) %>%
  filter(.metric == "roc_auc") %>%
  slice_max(mean, n = 1) %>%
  mutate(model = "KNN-Downsample")

rf_best <- purrr::map_dfr(names(rf_tunes), function(nm) {
  collect_metrics(rf_tunes[[nm]]) %>%
    filter(.metric == "roc_auc") %>%
    slice_max(mean, n = 1) %>%
    mutate(model = nm)
})

best_metrics <- bind_rows(knn_best, rf_best)
print(best_metrics %>% select(model, mean, std_err))

# Plot comparison
comparison_plot <- best_metrics %>%
  ggplot(aes(x = model, y = mean, ymin = mean - std_err, ymax = mean + std_err)) +
  geom_point(size = 3) +
  geom_errorbar(width = 0.2) +
  labs(title = "Model Comparison: ROC AUC", y = "Mean ROC AUC", x = NULL) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(comparison_plot)

# Select best model
best_model_id <- best_metrics %>% slice_max(mean, n = 1) %>% pull(model)
cat("Best model:", best_model_id, "\n")

# 14. Final fit
cat("Creating final model fit...\n")
if (best_model_id == "KNN-Downsample") {
  best_workflow   <- wf_knn_downsample
  tuning_results  <- tune_knn_down
} else {
  rf_name         <- best_model_id
  best_workflow   <- rf_workflows[[rf_name]]
  tuning_results  <- rf_tunes[[rf_name]]
}

best_params <- select_best(tuning_results, "roc_auc")
print(best_params)

final_wf    <- finalize_workflow(best_workflow, best_params)
final_fit   <- last_fit(final_wf, split_obj)

# 15. Model evaluation
cat("Evaluating final model...\n")
metrics     <- collect_metrics(final_fit)
print(metrics)

test_preds  <- collect_predictions(final_fit)
cm          <- conf_mat(test_preds, truth = default, estimate = .pred_class)
print(cm)
autoplot(cm, type = "heatmap") + labs(title = "Confusion Matrix Heatmap")

# 16. ROC & Precision-Recall Curves
roc_curve   <- test_preds %>%
  roc_curve(truth = default, .pred_default) %>%
  autoplot() + labs(title = "ROC Curve")
pr_curve    <- test_preds %>%
  pr_curve(truth = default, .pred_default) %>%
  autoplot() + labs(title = "Precision-Recall Curve")
print(roc_curve); print(pr_curve)

# 17. Threshold Optimization for F1/F2
cat("Optimizing classification threshold...\n")
thresholds      <- seq(0.1, 0.9, by = 0.05)
threshold_metrics <- purrr::map_dfr(thresholds, function(t) {
  test_preds %>%
    mutate(
      .pred_class_t = factor(if_else(.pred_default >= t, "default", "no_default"),
                             levels = levels(default))
    ) %>%
    metrics(truth = default, estimate = .pred_class_t) %>%
    select(.metric, .estimator, .estimate) %>%
    bind_cols(threshold = t)
})

f_metrics      <- threshold_metrics %>%
  filter(.metric %in% c("sens", "spec", "accuracy", "precision", "recall")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(
    f1_score = 2 * (precision * recall) / (precision + recall),
    f2_score = 5 * (precision * recall) / (4 * precision + recall)
  )

opt_f1_threshold <- f_metrics %>% slice_max(f1_score, n = 1) %>% pull(threshold)
opt_f2_threshold <- f_metrics %>% slice_max(f2_score, n = 1) %>% pull(threshold)
cat("Optimal threshold (F1):", opt_f1_threshold, "\n")
cat("Optimal threshold (F2):", opt_f2_threshold, "\n")

threshold_plot <- f_metrics %>%
  pivot_longer(cols = c(sens, spec, accuracy, precision, recall, f1_score, f2_score),
               names_to = "metric", values_to = "value") %>%
  ggplot(aes(x = threshold, y = value, color = metric)) +
  geom_line() +
  geom_vline(xintercept = opt_f2_threshold, linetype = "dashed") +
  annotate("text", x = opt_f2_threshold + 0.05, y = 0.5,
           label = paste("F2 optimal =", opt_f2_threshold), angle = 90) +
  labs(title = "Metrics by Classification Threshold",
       y = "Metric Value", x = "Threshold") +
  theme_bw()
print(threshold_plot)

# Final predictions with optimized threshold
final_preds  <- test_preds %>%
  mutate(.pred_class_opt = factor(if_else(.pred_default >= opt_f2_threshold,
                                          "default", "no_default"),
                                  levels = levels(default)))
cm_opt       <- conf_mat(final_preds, truth = default, estimate = .pred_class_opt)
print(cm_opt)

final_metrics <- bind_rows(
  test_preds %>% metrics(truth = default, estimate = .pred_class) %>% mutate(threshold = "default (0.5)"),
  final_preds %>% metrics(truth = default, estimate = .pred_class_opt) %>%
    mutate(threshold = paste0("optimized (", opt_f2_threshold, ")"))
)
print(final_metrics)

# 18. Variable Importance
cat("Calculating variable importance...\n")
if (grepl("RF", best_model_id)) {
  vip_plot <- vip(final_fit$.workflow[[1]], num_features = 10) +
    labs(title = "Top 10 Variable Importance")
  print(vip_plot)
} else {
  tryCatch({
    vip_plot <- vip(final_fit$.workflow[[1]], num_features = 10) +
      labs(title = "Top 10 Variable Importance")
    print(vip_plot)
  }, error = function(e) {
    cat("Variable importance not available for kNN\n")
  })
}

# 19. Model Explanation with LIME
cat("Generating LIME explanations...\n")
set.seed(2025)
explanation_cases <- test_data %>%
  group_by(default) %>%
  slice_sample(n = 2) %>%
  ungroup()
explainer <- lime(train_data %>% select(-default), final_fit$.workflow[[1]])
explanation <- explain(
  explanation_cases %>% select(-default),
  explainer,
  n_features = 6,
  n_permutations = 500
)
lime_plot <- plot_explanations(explanation) +
  labs(title = "LIME Explanations for Sample Cases")
print(lime_plot)

# 20. Cost-Benefit Analysis
cat("Assessing business impact...\n")
cost_fn <- 1000; cost_fp <- 100
cost_analysis <- purrr::map_dfr(thresholds, function(t) {
  preds_t <- test_preds %>%
    mutate(.pred_class_t = factor(if_else(.pred_default >= t, "default", "no_default"),
                                  levels = levels(default)))
  cm_t <- table(preds_t$.pred_class_t, preds_t$default)
  if (all(dim(cm_t) == c(2,2))) {
    fp <- cm_t["default","no_default"]
    fn <- cm_t["no_default","default"]
    total_cost <- fp*cost_fp + fn*cost_fn
  } else {
    fp <- fn <- NA_real_; total_cost <- NA_real_
  }
  tibble(threshold = t, false_positives = fp,
         false_negatives = fn, total_cost = total_cost)
})
optimal_cost_threshold <- cost_analysis %>% slice_min(total_cost, n = 1) %>% pull(threshold)
cat("Cost-minimizing threshold:", optimal_cost_threshold, "\n")

cost_plot <- cost_analysis %>%
  ggplot(aes(x = threshold)) +
  geom_line(aes(y = false_positives, color = "False Positives")) +
  geom_line(aes(y = false_negatives, color = "False Negatives")) +
  geom_line(aes(y = total_cost/1000, color = "Total Cost (thousands)")) +
  geom_vline(xintercept = optimal_cost_threshold, linetype = "dashed") +
  annotate("text", x = optimal_cost_threshold + 0.05,
           y = max(cost_analysis$false_positives, na.rm=TRUE)/2,
           label = paste("Cost optimal =", optimal_cost_threshold),
           angle = 90) +
  labs(title = "Cost Analysis by Threshold",
       y = "Count / Cost", x = "Threshold") +
  theme_bw()
print(cost_plot)

# 21. Final Recommendations
cat("\nFINAL RECOMMENDATIONS:\n")
cat("Best model: ", best_model_id, "\n")
cat("Best parameters: ",
    paste(names(best_params), "=", unlist(best_params), collapse = ", "), "\n")
cat("Thresholds:\n")
cat("- F1 optimal: ", opt_f1_threshold, "\n")
cat("- F2 optimal: ", opt_f2_threshold, "\n")
cat("- Cost-minimal: ", optimal_cost_threshold, "\n")
cat("\nFrom a business perspective, use the cost-minimizing threshold of",
    optimal_cost_threshold, "in production.\n")

# 22. Save Final Model and Reports
cat("\nSaving final model and results...\n")
final_results <- list(
  workflow          = final_fit$.workflow[[1]],
  optimal_threshold = optimal_cost_threshold,
  metrics           = final_metrics,
  variable_importance = if (exists("vip_plot")) vip_plot else NULL
)
saveRDS(final_results,
        paste0("final_", gsub("[- ]", "_", tolower(best_model_id)), "_model.rds"))

# 23. Clean Up
plan(sequential)
toc()  # Total execution time
cat("Optimized modeling pipeline complete.\n")

