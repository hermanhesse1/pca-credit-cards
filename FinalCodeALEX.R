# ------------------------------
# 1. Load Libraries & Set Seed
# ------------------------------
library(tidymodels)    # unified modeling framework
library(themis)        # for downsampling
library(doParallel)    # parallel backend
library(readr)         # fast CSV reading
library(janitor)       # clean_names()
set.seed(123)

# -----------------------------------
# 2. Read, Clean & Preprocess Data
# -----------------------------------
data <- read_csv("ohe28_cleaned_data.csv", show_col_types = FALSE) %>%
  clean_names()

if (!"default" %in% names(data)) {
  stop("Column 'default' not found. Available columns: ",
       paste(names(data), collapse = ", "))
}

data <- data %>%
  mutate(default = factor(default,
                          levels = c(0, 1),
                          labels = c("no", "yes"))) %>%
  filter(!is.na(default))  # drop any NA labels

# -----------------------------
# 3. Train/Test Split
# -----------------------------
split      <- initial_split(data, prop = 0.8, strata = default)
train_data <- training(split)
test_data  <- testing(split)

# -----------------------------
# 4. Define Preprocessing Recipes
# -----------------------------
pca_variance <- 0.90

knn_recipe <- recipe(default ~ ., data = train_data) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_pca(all_numeric_predictors(), threshold = pca_variance) %>%
  step_downsample(default)

rf_recipe <- recipe(default ~ ., data = train_data) %>%
  step_zv(all_predictors())

# -----------------------------
# 5. Model Specifications
# -----------------------------
knn_spec <- nearest_neighbor(
  mode       = "classification",
  neighbors  = tune(),
  dist_power = 2
) %>%
  set_engine("kknn")

rf_spec <- rand_forest(
  mode   = "classification",
  mtry   = tune(),
  min_n  = tune(),
  trees  = 500
) %>%
  set_engine("ranger", importance = "impurity")

# -----------------------------
# 6. Workflows
# -----------------------------
knn_wf <- workflow() %>%
  add_model(knn_spec) %>%
  add_recipe(knn_recipe)

rf_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(rf_recipe)

# -----------------------------
# 7. Cross‐Validation & Grids
# -----------------------------
cv_folds <- vfold_cv(train_data, v = 5, strata = default)

knn_grid <- tibble(neighbors = seq(3, 25, by = 2))

p       <- ncol(train_data) - 1
rf_grid <- expand.grid(
  mtry  = round(seq(sqrt(p), min(p, 2 * sqrt(p)), length.out = 5)),
  min_n = c(2, 5, 10, 20, 50)
)

# -----------------------------
# 8. Parallel Tuning with Caching
# -----------------------------
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

metrics <- metric_set(
  roc_auc,
  yardstick::accuracy,
  yardstick::sensitivity,
  yardstick::specificity
)

if (!file.exists("knn_tune.rds")) {
  knn_tune <- tune_grid(
    knn_wf,
    resamples = cv_folds,
    grid      = knn_grid,
    metrics   = metrics,
    control   = control_grid(save_pred = TRUE)
  )
  saveRDS(knn_tune, "knn_tune.rds")
} else {
  knn_tune <- readRDS("knn_tune.rds")
}

if (!file.exists("rf_tune.rds")) {
  rf_tune <- tune_grid(
    rf_wf,
    resamples = cv_folds,
    grid      = rf_grid,
    metrics   = metrics,
    control   = control_grid(save_pred = TRUE)
  )
  saveRDS(rf_tune, "rf_tune.rds")
} else {
  rf_tune <- readRDS("rf_tune.rds")
}

stopCluster(cl)

# -----------------------------
# 9. Select Best Hyperparameters
# -----------------------------
knn_best <- select_best(knn_tune, metric = "roc_auc")
rf_best  <- select_best(rf_tune,  metric = "roc_auc")

# -----------------------------
# 10. Finalize & Fit Final Models
# -----------------------------
final_knn_wf <- finalize_workflow(knn_wf, knn_best)
final_rf_wf  <- finalize_workflow(rf_wf,  rf_best)

final_knn_fit <- fit(final_knn_wf, data = train_data)
final_rf_fit  <- fit(final_rf_wf,  data = train_data)

# -----------------------------
# 11. Test‐Set Predictions
# -----------------------------
knn_preds <- predict(final_knn_fit, test_data, type = "prob")
rf_preds  <- predict(final_rf_fit,  test_data, type = "prob")

knn_res <- test_data %>%
  select(default) %>%
  bind_cols(knn_preds) %>%
  mutate(
    .pred_class_05 = factor(if_else(.pred_yes >= 0.5, "yes", "no"),
                            levels = c("no", "yes"))
  )

rf_res <- test_data %>%
  select(default) %>%
  bind_cols(rf_preds) %>%
  mutate(
    .pred_class_05 = factor(if_else(.pred_yes >= 0.5, "yes", "no"),
                            levels = c("no", "yes"))
  )

# -----------------------------
# 12. Compute Core Metrics
# -----------------------------
library(yardstick)  # for ppv() and sens()

core_metrics <- bind_rows(
  knn_res %>% roc_auc(truth = default, .pred_yes)              %>% mutate(model="KNN"),
  rf_res  %>% roc_auc(truth = default, .pred_yes)              %>% mutate(model="RF"),
  knn_res %>% accuracy(truth = default, .pred_class_05)        %>% mutate(model="KNN"),
  rf_res  %>% accuracy(truth = default, .pred_class_05)        %>% mutate(model="RF"),
  knn_res %>% ppv(truth = default, estimate = .pred_class_05,
                  event_level = "second")                     %>% mutate(model="KNN"),
  rf_res  %>% ppv(truth = default, estimate = .pred_class_05,
                  event_level = "second")                     %>% mutate(model="RF"),
  knn_res %>% sens(truth = default, estimate = .pred_class_05,
                   event_level = "second")                    %>% mutate(model="KNN"),
  rf_res  %>% sens(truth = default, estimate = .pred_class_05,
                   event_level = "second")                    %>% mutate(model="RF")
)
print(core_metrics)

# -----------------------------
# 13. Cost‐Based Threshold Sweep
# -----------------------------
cost_fp    <- 1; cost_fn <- 5
thresholds <- seq(0, 1, by = 0.01)

compute_cost <- function(probs, actual, th) {
  pred <- if_else(probs >= th, "yes", "no")
  fp   <- sum(pred == "yes" & actual == "no")
  fn   <- sum(pred == "no"  & actual == "yes")
  cost_fp * fp + cost_fn * fn
}

knn_costs  <- sapply(thresholds, compute_cost, probs = knn_res$.pred_yes, actual = knn_res$default)
knn_opt_th <- thresholds[which.min(knn_costs)]
cat("KNN optimal threshold:", knn_opt_th, "\n")

rf_costs   <- sapply(thresholds, compute_cost, probs = rf_res$.pred_yes,  actual = rf_res$default)
rf_opt_th  <- thresholds[which.min(rf_costs)]
cat("RF optimal threshold:", rf_opt_th, "\n")

# -----------------------------
# 14. Confusion & Final Metrics
# -----------------------------
# KNN @ optimal
knn_res_opt <- knn_res %>%
  mutate(.pred_opt = factor(if_else(.pred_yes >= knn_opt_th, "yes", "no"),
                            levels = c("no","yes")))

print(conf_mat(knn_res_opt, truth=default, estimate=.pred_opt))
knn_final <- tibble(
  accuracy  = accuracy(knn_res_opt, truth=default, estimate=.pred_opt)$.estimate,
  precision = ppv(knn_res_opt, truth=default, estimate=.pred_opt,
                  event_level="second")$.estimate,
  recall    = sens(knn_res_opt, truth=default, estimate=.pred_opt,
                   event_level="second")$.estimate
)
cat("KNN @", knn_opt_th, "\n"); print(knn_final)

# RF @ optimal
rf_res_opt <- rf_res %>%
  mutate(.pred_opt = factor(if_else(.pred_yes >= rf_opt_th, "yes","no"),
                            levels = c("no","yes")))

print(conf_mat(rf_res_opt, truth=default, estimate=.pred_opt))
rf_final <- tibble(
  accuracy  = accuracy(rf_res_opt, truth=default, estimate=.pred_opt)$.estimate,
  precision = ppv(rf_res_opt, truth=default, estimate=.pred_opt,
                  event_level="second")$.estimate,
  recall    = sens(rf_res_opt, truth=default, estimate=.pred_opt,
                   event_level="second")$.estimate
)
cat("RF @", rf_opt_th, "\n"); print(rf_final)


