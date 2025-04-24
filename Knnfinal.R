# ===== Optimized Credit Default Classification with Tidymodels =====

# 0. Install missing engine if needed
if (!rlang::is_installed("ranger")) install.packages("ranger")

# 1. Load Libraries
library(tidymodels)  # recipes, parsnip, workflows, tune, yardstick
library(themis)      # step_downsample()
library(vip)         # variable importance
library(doFuture)    # parallel backend
library(finetune)    # tune_race_anova()
library(lime)        # LIME explanations
library(probably)    # threshold utilities
library(tictoc)      # timing
library(ggplot2)     # plots
library(dplyr)       # data manipulation
library(purrr)       # map functions

# 2. Parallel Processing Setup
registerDoFuture()
plan(multisession, workers = max(1, parallel::detectCores() - 1))
tic("Total Execution Time")

# 3. Load & Clean Data
data_raw <- readr::read_csv("ohe28_cleaned_data.csv", show_col_types = FALSE) %>%
  janitor::clean_names() %>%
  mutate(default = factor(default, levels = c(0, 1),
                          labels = c("no_default", "default")))

# 4. Train/Test Split
set.seed(2025)
split_obj  <- initial_split(data_raw, prop = 0.8, strata = default)
train_data <- training(split_obj)
test_data  <- testing(split_obj)

# 5. Define Recipes
base_recipe <- recipe(default ~ ., data = train_data) %>%
  step_zv(all_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.8) %>%
  step_normalize(all_numeric_predictors())

recipe_down <- base_recipe %>%
  step_downsample(default, under_ratio = tune()) %>%
  step_pca(all_numeric_predictors(), threshold = tune())

recipe_nobal <- base_recipe %>%
  step_pca(all_numeric_predictors(), threshold = tune())

# 6. Model Specifications
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

# 7. Workflows
wf_knn <- workflow() %>%
  add_model(knn_spec) %>%
  add_recipe(recipe_down)

rf_wfs <- map(rf_specs, ~ workflow() %>%
                add_model(.x) %>%
                add_recipe(recipe_nobal))
names(rf_wfs) <- names(rf_specs)

# 8. Resampling & Grids
set.seed(2025)
folds <- vfold_cv(train_data, v = 5, strata = default)

knn_grid <- grid_latin_hypercube(
  neighbors(range = c(3, 20)),
  weight_func(values = c("rectangular", "triangular", "epanechnikov", "gaussian")),
  dist_power(range = c(1, 2)),
  threshold(range = c(0.7, 0.95)),    # PCA variance
  under_ratio(range = c(1.5, 3)),     # down‐sample ratio
  size = 15
)

rf_grid <- grid_latin_hypercube(
  mtry(range = c(5, 20)),
  min_n(range = c(2, 10)),
  threshold(range = c(0.7, 0.95)),    # PCA variance
  size = 15
)

# 9. Control & Metrics
ctrl <- control_race(
  save_pred     = TRUE,
  save_workflow = FALSE,    # ← turn off saving big workflows
  parallel_over = "resamples",
  allow_par     = TRUE,
  verbose       = TRUE
)
model_metrics <- metric_set(roc_auc, accuracy, pr_auc, sens, spec)

# 10. Hyperparameter Tuning
tic("KNN tuning")
tuned_knn <- tune_race_anova(
  wf_knn, resamples = folds, grid = knn_grid,
  metrics = model_metrics, control = ctrl
)
toc()

tuned_rf <- list()
for (nm in names(rf_wfs)) {
  tic(paste("RF tuning:", nm))
  tuned_rf[[nm]] <- tune_race_anova(
    rf_wfs[[nm]], resamples = folds, grid = rf_grid,
    metrics = model_metrics, control = ctrl
  )
  toc()
}

# 11. Compare Models
knn_best <- collect_metrics(tuned_knn) %>%
  filter(.metric == "roc_auc") %>%
  slice_max(mean, n = 1) %>%
  mutate(model = "KNN")

rf_best <- map_dfr(names(tuned_rf), function(nm) {
  collect_metrics(tuned_rf[[nm]]) %>%
    filter(.metric == "roc_auc") %>%
    slice_max(mean, n = 1) %>%
    mutate(model = nm)
})

best_metrics <- bind_rows(knn_best, rf_best)
print(best_metrics %>% select(model, mean, std_err))

best_metrics %>%
  ggplot(aes(model, mean, ymin = mean - std_err, ymax = mean + std_err)) +
  geom_point(size = 3) +
  geom_errorbar(width = 0.2) +
  labs(title = "Model Comparison: ROC AUC") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

best_model <- best_metrics$model[which.max(best_metrics$mean)]
cat("Best model is:", best_model, "\n")

# 12. Final Fit
if (best_model == "KNN") {
  final_wf   <- finalize_workflow(wf_knn, select_best(tuned_knn, metric = "roc_auc"))
} else {
  final_wf   <- finalize_workflow(rf_wfs[[best_model]], select_best(tuned_rf[[best_model]], metric = "roc_auc"))
}

final_res <- last_fit(final_wf, split_obj)

# 13. Final Evaluation
final_metrics <- collect_metrics(final_res)
print(final_metrics)

test_preds <- collect_predictions(final_res)
cm <- conf_mat(test_preds, truth = default, estimate = .pred_class)
print(cm)
autoplot(cm, type = "heatmap") + labs(title = "Final Confusion Matrix")

roc_plot <- test_preds %>%
  roc_curve(default, .pred_default) %>%
  autoplot() + labs(title = "ROC Curve")

pr_plot <- test_preds %>%
  pr_curve(default, .pred_default) %>%
  autoplot() + labs(title = "Precision-Recall Curve")

print(roc_plot); print(pr_plot)

# 14. Threshold & Business Analysis
thresholds <- seq(0.1, 0.9, by = 0.05)

threshold_metrics <- map_dfr(thresholds, function(t) {
  test_preds %>%
    mutate(.pred_class_t = factor(if_else(.pred_default >= t, "default", "no_default"),
                                  levels = levels(default))) %>%
    metrics(truth = default, estimate = .pred_class_t) %>%
    select(.metric, .estimate) %>%
    bind_cols(threshold = t)
})

f_metrics <- threshold_metrics %>%
  filter(.metric %in% c("precision", "recall")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(
    f1_score = 2 * (precision * recall) / (precision + recall),
    f2_score = 5 * (precision * recall) / (4 * precision + recall),
    threshold = thresholds
  )

opt_f1_threshold <- f_metrics %>% slice_max(f1_score, n = 1) %>% pull(threshold)
opt_f2_threshold <- f_metrics %>% slice_max(f2_score, n = 1) %>% pull(threshold)

cost_fn <- 1000
cost_fp <- 100
cost_analysis <- map_dfr(thresholds, function(t) {
  preds_t <- test_preds %>%
    mutate(.pred_class_t = factor(if_else(.pred_default >= t, "default", "no_default"),
                                  levels = levels(default)))
  cm_t <- table(preds_t$.pred_class_t, preds_t$default)
  if (all(dim(cm_t) == c(2, 2))) {
    fp <- cm_t["default", "no_default"]
    fn <- cm_t["no_default", "default"]
    total_cost <- fp * cost_fp + fn * cost_fn
  } else {
    fp <- fn <- NA_real_
    total_cost <- NA_real_
  }
  tibble(threshold = t,
         false_positives = fp,
         false_negatives = fn,
         total_cost = total_cost)
})

optimal_cost_threshold <- cost_analysis %>%
  filter(!is.na(total_cost)) %>%
  slice_min(total_cost, n = 1) %>%
  pull(threshold)

# 15. Print Thresholds
cat("\nThresholds:\n")
cat("- F1 optimal: ", opt_f1_threshold, "\n")
cat("- F2 optimal: ", opt_f2_threshold, "\n")
cat("- Cost-minimal: ", optimal_cost_threshold, "\n")
cat("\nUse cost-minimizing threshold of", optimal_cost_threshold, "in production.\n")

# 16. Plots for Threshold Analysis
threshold_metrics %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(threshold, .estimate)) +
  geom_line() +
  labs(title = "Accuracy by Threshold", y = "Accuracy", x = "Threshold") +
  theme_bw()

f_metrics %>%
  pivot_longer(c(f1_score, f2_score), names_to = "metric", values_to = "value") %>%
  ggplot(aes(threshold, value, color = metric)) +
  geom_line() +
  labs(title = "F1 & F2 by Threshold", y = "Score", x = "Threshold") +
  theme_bw()

cost_analysis %>%
  ggplot(aes(threshold)) +
  geom_line(aes(y = total_cost), color = "black") +
  geom_vline(xintercept = optimal_cost_threshold, linetype = "dashed") +
  labs(title = "Total Cost by Threshold", y = "Total Cost", x = "Threshold") +
  theme_bw()

# 17. Save Final Model & Results
final_results <- list(
  workflow           = final_wf,
  optimal_threshold  = optimal_cost_threshold,
  metrics            = final_metrics,
  variable_importance = if (exists("vip_plot")) vip_plot else NULL
)
saveRDS(final_results,
        paste0("final_", gsub("[- ]", "_", tolower(best_model)), "_model.rds"))

# 18. Clean Up
plan(sequential)
toc()

final_acc <- final_metrics %>%
  filter(.metric == "accuracy") %>%
  pull(.estimate)
cat("Final test-set accuracy:", round(final_acc, 4), "\n")

# —— Step 19: Precision & Recall at the Optimal Threshold ——

library(dplyr)
library(yardstick)

# Step 12: finalize & last_fit
final_res <- last_fit(final_wf, split_obj)

# Step 13: collect predictions
test_preds <- collect_predictions(final_res)

thresholds <- seq(0.1, 0.9, by = 0.05)

cost_fn <- 1000
cost_fp <- 100

cost_analysis <- purrr::map_dfr(thresholds, function(t) {
  preds_t <- test_preds %>%
    mutate(.pred_class_t = factor(
      if_else(.pred_default >= t, "default", "no_default"),
      levels = levels(default)
    ))
  cm_t <- table(preds_t$.pred_class_t, preds_t$default)
  fp <- cm_t["default", "no_default"]
  fn <- cm_t["no_default", "default"]
  tibble(
    threshold       = t,
    false_positives = fp,
    false_negatives = fn,
    total_cost      = fp * cost_fp + fn * cost_fn
  )
})

optimal_cost_threshold <- cost_analysis %>%
  filter(!is.na(total_cost)) %>%
  slice_min(total_cost, n = 1) %>%
  pull(threshold)

library(dplyr)
library(yardstick)
library(dplyr)
library(yardstick)

# ─── 1. Binarize at your optimal threshold ───────────────────────────────────────
# Make sure you’ve already run:
#   test_preds <- collect_predictions(final_res)
#   optimal_cost_threshold <- <your 0.10 value>
preds_opt <- test_preds %>%
  mutate(.pred_class_opt = factor(
    if_else(.pred_default >= optimal_cost_threshold, "default", "no_default"),
    levels = c("no_default", "default")
  ))

# ─── 2. Build the confusion matrix ───────────────────────────────────────────────
cm_opt <- conf_mat(preds_opt, truth = default, estimate = .pred_class_opt)
print(cm_opt)

# You can also see it as a table:
tbl <- cm_opt$table
#              Truth
# Prediction   no_default  default
#  no_default      TN         FN
#  default         FP         TP

TN <- tbl["no_default", "no_default"]
FP <- tbl["default",    "no_default"]
FN <- tbl["no_default", "default"]
TP <- tbl["default",    "default"]
N  <- TN + FP + FN + TP

# ─── 3. Compute the metrics by formula ──────────────────────────────────────────
accuracy  <- (TP + TN) / N
precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)

# ─── 4. Print them neatly ───────────────────────────────────────────────────────
cat(sprintf(
  "At threshold = %.2f:\n  • Accuracy : %.3f\n  • Precision: %.3f\n  • Recall   : %.3f\n",
  optimal_cost_threshold,
  accuracy, precision, recall
))


