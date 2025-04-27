# ------------------------------
# 1. Load Libraries & Set Seed
# ------------------------------
library(tidymodels)    # modelling + yardstick, etc.
library(themis)        # downsampling
library(doParallel)    # parallel backend
library(readr)         # fast CSV reading
library(janitor)       # clean_names()

set.seed(123)

# -----------------------------------
# 2. Read, Clean & Preprocess Data
# -----------------------------------
data <- read_csv("ohe28_cleaned_data.csv", show_col_types = FALSE) %>%
  clean_names()

if (!"default" %in% names(data)) stop("Column 'default' not found.")

data <- data %>%
  mutate(
    # Map 1 → “yes”, 0 → “no” and put “yes” first so it's the positive class
    default = factor(default,
                     levels = c(1, 0),
                     labels = c("yes", "no"))
  ) %>%
  filter(!is.na(default))

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
) %>% set_engine("kknn")

rf_spec <- rand_forest(
  mode   = "classification",
  mtry   = tune(),
  min_n  = tune(),
  trees  = 500
) %>% set_engine("ranger", importance = "impurity")

# -----------------------------
# 6. Workflows
# -----------------------------
knn_wf <- workflow() %>% add_model(knn_spec) %>% add_recipe(knn_recipe)
rf_wf  <- workflow() %>% add_model(rf_spec)  %>% add_recipe(rf_recipe)

# -----------------------------
# 7. Cross‐Validation & Grids
# -----------------------------
cv_folds <- vfold_cv(train_data, v = 5, strata = default)

knn_grid <- tibble(neighbors = seq(3, 25, by = 2))
p        <- ncol(train_data) - 1
rf_grid  <- expand.grid(
  mtry  = round(seq(sqrt(p), min(p, 2 * sqrt(p)), length.out = 5)),
  min_n = c(2, 5, 10, 20, 50)
)

# -----------------------------
# 8. Parallel Tuning with Caching
# -----------------------------
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# ONLY class–prob metrics here to satisfy “only numeric metrics”
metrics <- metric_set(
  roc_auc,
  pr_auc
)

knn_tune <- tune_grid(
  knn_wf,
  resamples = cv_folds,
  grid      = knn_grid,
  metrics   = metrics,
  control   = control_grid(save_pred = TRUE)
)
saveRDS(knn_tune, "knn_tune.rds")

rf_tune <- tune_grid(
  rf_wf,
  resamples = cv_folds,
  grid      = rf_grid,
  metrics   = metrics,
  control   = control_grid(save_pred = TRUE)
)
saveRDS(rf_tune, "rf_tune.rds")

stopCluster(cl)

# -----------------------------
# 9. Select Best Hyperparameters
# -----------------------------
knn_best <- select_best(knn_tune, metric = "roc_auc")
rf_best  <- select_best(rf_tune,  metric = "roc_auc")

# -----------------------------
# 10. Finalize & Fit Final Models
# -----------------------------
final_knn_fit <- finalize_workflow(knn_wf, knn_best) %>% fit(data = train_data)
final_rf_fit  <- finalize_workflow(rf_wf,  rf_best)  %>% fit(data = train_data)

# -----------------------------
# 11. Test‐Set Predictions
# -----------------------------
knn_res <- test_data %>%
  bind_cols(predict(final_knn_fit, test_data, type = "prob")) %>%
  bind_cols(predict(final_knn_fit, test_data, type = "class")) %>%
  rename(.pred_class_05 = .pred_class)

rf_res  <- test_data %>%
  bind_cols(predict(final_rf_fit, test_data, type = "prob")) %>%
  bind_cols(predict(final_rf_fit, test_data, type = "class")) %>%
  rename(.pred_class_05 = .pred_class)

# -----------------------------
# 12. Compute Core Metrics & Print
# -----------------------------
library(yardstick)

core_metrics <- bind_rows(
  # ROC AUC & PR AUC
  knn_res %>% roc_auc(truth = default, .pred_yes)   %>% mutate(model = "KNN"),
  rf_res  %>% roc_auc(truth = default, .pred_yes)   %>% mutate(model = "RF"),
  knn_res %>% pr_auc(truth = default, .pred_yes)    %>% mutate(model = "KNN"),
  rf_res  %>% pr_auc(truth = default, .pred_yes)    %>% mutate(model = "RF"),
  # Class metrics
  knn_res %>% accuracy(truth = default, .pred_class_05)                     %>% mutate(model="KNN"),
  rf_res  %>% accuracy(truth = default, .pred_class_05)                     %>% mutate(model="RF"),
  knn_res %>% sensitivity(truth = default, .pred_class_05)                  %>% mutate(model="KNN"),
  rf_res  %>% sensitivity(truth = default, .pred_class_05)                  %>% mutate(model="RF"),
  knn_res %>% specificity(truth = default, .pred_class_05)                  %>% mutate(model="KNN"),
  rf_res  %>% specificity(truth = default, .pred_class_05)                  %>% mutate(model="RF"),
  knn_res %>% precision(truth = default, estimate = .pred_class_05)         %>% mutate(model="KNN"),
  rf_res  %>% precision(truth = default, estimate = .pred_class_05)         %>% mutate(model="RF"),
  knn_res %>% recall(truth = default, estimate = .pred_class_05)            %>% mutate(model="KNN"),
  rf_res  %>% recall(truth = default, estimate = .pred_class_05)            %>% mutate(model="RF")
)

print(core_metrics)

# -----------------------------
# 13. Confusion Matrices & Optimized Metrics
# -----------------------------
cost_fp    <- 1; cost_fn <- 5
thresholds <- seq(0, 1, by = 0.01)

compute_cost <- function(probs, actual, th) {
  pred <- if_else(probs >= th, "yes", "no")
  fp   <- sum(pred=="yes" & actual=="no")
  fn   <- sum(pred=="no"  & actual=="yes")
  cost_fp*fp + cost_fn*fn
}

knn_costs  <- sapply(thresholds, compute_cost, probs = knn_res$.pred_yes, actual = knn_res$default)
knn_opt   <- thresholds[which.min(knn_costs)]
rf_costs  <- sapply(thresholds, compute_cost, probs = rf_res$.pred_yes,  actual = rf_res$default)
rf_opt    <- thresholds[which.min(rf_costs)]

knn_res_opt <- knn_res %>%
  mutate(.pred_opt = factor(if_else(.pred_yes >= knn_opt, "yes","no"),
                            levels = c("yes","no")))
rf_res_opt  <- rf_res  %>%
  mutate(.pred_opt = factor(if_else(.pred_yes >= rf_opt, "yes","no"),
                            levels = c("yes","no")))

cat("KNN Confusion Matrix @", knn_opt, "\n")
print(conf_mat(knn_res_opt, truth = default, estimate = .pred_opt))
cat("\nRF Confusion Matrix @", rf_opt, "\n")
print(conf_mat(rf_res_opt, truth = default, estimate = .pred_opt))

# -----------------------------
# 14. ROC Plot
# -----------------------------
library(ggplot2)

roc_data <- bind_rows(
  roc_curve(knn_res, truth = default, .pred_yes)   %>% mutate(model="KNN"),
  roc_curve(rf_res,  truth = default, .pred_yes)   %>% mutate(model="Random Forest")
)

auc_vals <- core_metrics %>%
  filter(.metric == "roc_auc") %>%
  select(model, .estimate)

ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path(size = 1.2) +
  geom_abline(linetype = "dashed", color = "grey50") +
  geom_text(data = auc_vals,
            aes(x = 0.6,
                y = ifelse(model == "KNN", 0.3, 0.2),
                label = paste0(model, ": AUC=", round(.estimate, 3))),
            hjust = 0) +
  labs(title = "ROC Curves for Final Models",
       x     = "False Positive Rate",
       y     = "True Positive Rate") +
  theme_minimal() +
  theme(legend.position = "bottom")


# -----------------------------
# 13b. Print Accuracy / Precision / Recall @ Optimal Threshold
# -----------------------------

# helper to pull the three metrics for a result‐set
get_core_class_metrics <- function(res, truth, estimate) {
  bind_rows(
    res %>% accuracy(truth = {{truth}}, estimate = {{estimate}}),
    res %>% precision(truth  = {{truth}}, estimate = {{estimate}}),
    res %>% recall(truth     = {{truth}}, estimate = {{estimate}})
  ) %>%
    select(.metric, .estimate)
}

# compute for each model and then tack on a model column
knn_metrics_opt <- get_core_class_metrics(knn_res_opt, default, .pred_opt) %>%
  mutate(model = "KNN") %>%
  select(model, .metric, .estimate)

rf_metrics_opt  <- get_core_class_metrics(rf_res_opt,  default, .pred_opt) %>%
  mutate(model = "RF") %>%
  select(model, .metric, .estimate)

cat("\nCore class‐metrics @ optimal thresholds:\n")
print(bind_rows(knn_metrics_opt, rf_metrics_opt))
