# 0. Install missing engine if needed
if (!rlang::is_installed("ranger")) install.packages("ranger")

# 1. Load Libraries
library(tidymodels)  # loads parsnip, recipes, workflows, tune, yardstick, etc.
library(yardstick)   # explicit for metrics
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
library(janitor)     # clean_names()

# 2. Parallel Processing Setup
registerDoFuture()
plan(multisession, workers = max(1, parallel::detectCores() - 1))
tic("Total Execution Time")

# 3. Load & Clean Data
data_raw <- readr::read_csv("ohe28_cleaned_data.csv", show_col_types = FALSE) %>%
  clean_names() %>%
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
  threshold(range = c(0.7, 0.95)),   
  under_ratio(range = c(1.5, 3)),    
  size = 15
)

rf_grid <- grid_latin_hypercube(
  mtry(range = c(5, 20)),
  min_n(range = c(2, 10)),
  threshold(range = c(0.7, 0.95)),   
  size = 15
)

# 9. Control & Metrics
ctrl <- control_race(
  save_pred     = TRUE,
  save_workflow = FALSE,
  parallel_over = "resamples",
  allow_par     = TRUE,
  verbose       = TRUE,
  burn_in       = 3
)
model_metrics <- metric_set(roc_auc, pr_auc, yardstick::accuracy, sensitivity, specificity)

# 10. Hyperparameter Tuning
tic("KNN tuning")
tuned_knn <- tune_race_anova(
  wf_knn,
  resamples = folds,
  grid      = knn_grid,
  metrics   = model_metrics,
  control   = ctrl
)
toc()

tuned_rf <- list()
for (nm in names(rf_wfs)) {
  tic(paste("RF tuning:", nm))
  tuned_rf[[nm]] <- tune_race_anova(
    rf_wfs[[nm]],
    resamples = folds,
    grid      = rf_grid,
    metrics   = model_metrics,
    control   = ctrl
  )
  toc()
}

# 11. Compare Models
knn_best <- collect_metrics(tuned_knn) %>%
  filter(.metric == "roc_auc") %>% slice_max(mean, n = 1) %>%
  mutate(model = "KNN")

rf_best <- map_dfr(names(tuned_rf), function(nm) {
  collect_metrics(tuned_rf[[nm]]) %>%
    filter(.metric == "roc_auc") %>% slice_max(mean, n = 1) %>%
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
  final_wf <- finalize_workflow(wf_knn, select_best(tuned_knn, metric = "roc_auc"))
} else {
  final_wf <- finalize_workflow(rf_wfs[[best_model]], select_best(tuned_rf[[best_model]], metric = "roc_auc"))
}

final_res <- last_fit(final_wf, split_obj)

# 13. Final Evaluation
final_metrics <- collect_metrics(final_res)
print(final_metrics)

test_preds <- collect_predictions(final_res)

# 14. Precision & Recall at Optimal Threshold
thresholds <- seq(0.1, 0.9, by = 0.05)
cost_fp <- 100; cost_fn <- 1000

cost_analysis <- map_dfr(thresholds, function(t) {
  preds_t <- test_preds %>%
    mutate(.pred = factor(if_else(.pred_default >= t, "default", "no_default"),
                          levels = levels(default)))
  cm <- table(preds_t$.pred, preds_t$default)
  if (all(dim(cm)==c(2,2))) {
    fp <- cm["default","no_default"]; fn <- cm["no_default","default"]
    cost <- fp*cost_fp + fn*cost_fn
  } else {
    fp <- fn <- NA; cost <- NA
  }
  tibble(threshold=t, false_positives=fp, false_negatives=fn, total_cost=cost)
})
opt_thresh <- cost_analysis %>% slice_min(total_cost, n=1) %>% pull(threshold)

preds_opt <- test_preds %>%
  mutate(.pred_opt = factor(if_else(.pred_default >= opt_thresh, "default", "no_default"),
                            levels = levels(default)))
cm_opt <- conf_mat(preds_opt, truth=default, estimate=.pred_opt)
print(cm_opt)

tbl <- cm_opt$table
TN <- tbl["no_default","no_default"]; FP <- tbl["default","no_default"]
FN <- tbl["no_default","default"]; TP <- tbl["default","default"]
accuracy  <- (TP+TN)/(TP+TN+FP+FN)
precision <- TP/(TP+FP)
recall    <- TP/(TP+FN)
cat(sprintf("At threshold=%.2f: Acc=%.3f, Prec=%.3f, Rec=%.3f\n",
            opt_thresh, accuracy, precision, recall))

# 15. Clean Up
plan(sequential)
toc()

