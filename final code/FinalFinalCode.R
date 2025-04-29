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

print(knn_best)
# or, to pull the raw number
knn_best$neighbors

library(tidyverse)
library(scales)

# 1. Load & clean (as before)
data <- read_csv("ohe28_cleaned_data.csv", show_col_types = FALSE) %>%
  clean_names() %>%
  mutate(
    default = factor(default,
                     levels = c(1, 0),
                     labels = c("yes", "no"))
  )

# 2. Boxplot of credit limit by default status
ggplot(data, aes(x = default, y = limit_bal, fill = default)) +
  geom_boxplot(outlier.alpha = 0.2) +
  scale_y_continuous(labels = comma, 
                     limits = c(0, 1e6),   # match your max = 1,000,000
                     breaks = seq(0, 1e6, by = 200e3)) +
  scale_fill_manual(values = c("yes" = "#D55E00", "no" = "#0072B2")) +
  labs(
    title = "Credit Limit Distribution by Default Status",
    subtitle = "Boxplots of LIMIT_BAL (NT$) for clients who default vs. not next month",
    x = "Default Next Month",
    y = "Credit Limit (NT$)"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")
install.packages("ggridges")
# 0. Load libraries
library(tidyverse)
library(scales)      # for comma() axis labels
library(ggridges)    # for ridgeline plots
library(patchwork)   # to combine multiple ggplots
library(broom)       # for tidying PCA results

# 1. Read & prepare
data <- read_csv("ohe28_cleaned_data.csv", show_col_types = FALSE) %>%
  clean_names() %>%
  mutate(
    # Make a human-readable factor
    default = factor(default,
                     levels = c(1, 0),
                     labels = c("Defaulted", "Did not default"))
  )

# 2. Basic bar chart: count of defaults vs non-defaults
p1 <- ggplot(data, aes(x = default, fill = default)) +
  geom_bar() +
  scale_fill_manual(values = c("Defaulted" = "#D55E00", 
                               "Did not default" = "#0072B2")) +
  labs(
    title = "Number of Clients by Next-Month Default Status",
    x     = "Next-Month Default?",
    y     = "Number of Clients"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# 3. Histogram: distribution of credit limits, overlaid by default status
p2 <- ggplot(data, aes(x = limit_bal, fill = default)) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 40) +
  scale_x_continuous(labels = comma, 
                     limits = c(0, 1e6), 
                     breaks = seq(0, 1e6, by = 200e3)) +
  scale_fill_manual(values = c("#D55E00", "#0072B2")) +
  labs(
    title = "Distribution of Credit Limits by Default Status",
    x     = "Credit Limit (NT Dollars)",
    y     = "Frequency",
    fill  = "Next-Month Default?"
  ) +
  theme_minimal(base_size = 14)

# 4. Violin + boxplot: credit limit by class, with outlier jitter
p3 <- ggplot(data, aes(x = default, y = limit_bal, fill = default)) +
  geom_violin(trim = FALSE, alpha = 0.4) +
  geom_boxplot(width = 0.2, outlier.shape = NA) +
  geom_jitter(width = 0.15, alpha = 0.1, color = "gray40") +
  scale_y_continuous(labels = comma) +
  scale_fill_manual(values = c("#D55E00", "#0072B2")) +
  labs(
    title = "Credit Limit by Default Status: Violin and Boxplot",
    x     = NULL,
    y     = "Credit Limit (NT Dollars)"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# 5. Ridgeline density: repayment status in the most recent month
#    (assuming PAY_0 is the variable for September status)
p4 <- ggplot(data, 
             aes(x = repayment_status_september_2005, 
                 y = default, 
                 fill = default)) +
  geom_density_ridges(alpha = 0.7, scale = 1.2) +
  scale_fill_manual(values = c("#D55E00", "#0072B2")) +
  labs(
    title = "Distribution of September 2005 Repayment Status by Default",
    x     = "Repayment Status (–2 = early, 0 = on time, 1+ = months late)",
    y     = "Next-Month Default?"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# 6. PCA scatter: first two components, colored by default
#    (This uses base prcomp + broom for tidying.)
numeric_data <- data %>% select(where(is.numeric))
pca_res <- prcomp(numeric_data, center = TRUE, scale. = TRUE)
pca_df  <- broom::augment(pca_res, data) %>% 
  select(.fittedPC1, .fittedPC2, default)

p5 <- ggplot(pca_df, aes(x = .fittedPC1, y = .fittedPC2, color = default)) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_manual(values = c("#D55E00", "#0072B2")) +
  labs(
    title = "PCA: Clients Projected onto First Two Components",
    x     = "Principal Component 1",
    y     = "Principal Component 2",
    color = "Next-Month Default?"
  ) +
  theme_minimal(base_size = 14)

# 7. Combine and display
(p1 | p2) / (p3 | p4)  

# If you want the PCA separately:
print(p5)


library(tidyverse)
library(scales)

data <- read_csv("ohe28_cleaned_data.csv", show_col_types = FALSE) %>%
  clean_names() %>%
  mutate(default = factor(default, levels = c(1, 0),
                          labels = c("Defaulted", "Did not default")))

ggplot(data, aes(x = default, fill = default)) +
  geom_bar(width = 0.6) +
  scale_fill_manual(values = c("Defaulted" = "#D55E00",
                               "Did not default" = "#0072B2")) +
  labs(
    title = "Number of Clients by Next-Month Default Status",
    x     = "Next-Month Default?",
    y     = "Count of Clients"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

ggplot(data, aes(x = limit_bal, fill = default)) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 40) +
  scale_x_continuous(labels = comma,
                     limits = c(0, 1e6),
                     breaks = seq(0, 1e6, by = 200e3)) +
  scale_fill_manual(values = c("#D55E00", "#0072B2")) +
  labs(
    title = "Distribution of Credit Limits by Default Status",
    x     = "Credit Limit (NT$)",
    y     = "Frequency",
    fill  = "Next-Month Default?"
  ) +
  theme_minimal(base_size = 14)


library(broom)

numeric_data <- data %>%
  select(where(is.numeric))

pca_res <- prcomp(numeric_data, center = TRUE, scale. = TRUE)

pca_df <- augment(pca_res, data) %>%
  select(.fittedPC1, .fittedPC2, default)

ggplot(pca_df, aes(x = .fittedPC1, y = .fittedPC2, color = default)) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_manual(values = c("Defaulted" = "#D55E00",
                                "Did not default" = "#0072B2")) +
  labs(
    title = "PCA Projection: Clients on the First Two Components",
    x     = "Principal Component 1",
    y     = "Principal Component 2",
    color = "Next-Month Default?"
  ) +
  theme_minimal(base_size = 14)

# 1. Define the “second‐best” KNN: Euclidean distance + triangular weighting
library(tidymodels)
library(ggplot2)
library(yardstick)

tri_knn_spec <- nearest_neighbor(
  mode       = "classification",
  neighbors  = tune(),
  dist_power = 2
) %>%
  set_engine("kknn", kernel = "triangular")

# 2. Reuse the same recipe & folds as before
tri_knn_wf <- workflow() %>%
  add_model(tri_knn_spec) %>%
  add_recipe(knn_recipe)

# 3. Tune on the 5-fold CV grid
tri_knn_tune <- tune_grid(
  tri_knn_wf,
  resamples = cv_folds,
  grid      = knn_grid,
  metrics   = metrics,
  control   = control_grid(save_pred = TRUE)
)

# 4. Select the best # of neighbors by ROC AUC
tri_knn_best <- select_best(tri_knn_tune, metric = "roc_auc")

# 5. Finalize & fit on full training set
final_tri_knn <- finalize_workflow(tri_knn_wf, tri_knn_best) %>%
  fit(data = train_data)

# 6. Predict on the test set & compute ROC curve
tri_knn_res <- test_data %>%
  bind_cols(predict(final_tri_knn, test_data, type = "prob")) %>%
  bind_cols(predict(final_tri_knn, test_data, type = "class")) %>%
  rename(.pred_class_tri = .pred_class)

tri_roc <- roc_curve(tri_knn_res, truth = default, .pred_yes) %>%
  mutate(model = paste0("Triangular KNN (k=", tri_knn_best$neighbors, ")"))

# 7. (Optional) Combine with unweighted KNN ROC data if you have it:
# combined_roc <- bind_rows(
#   roc_data,        # from unweighted KNN & RF
#   tri_roc
# )

# 8. Plot ROC curve for the triangular‐weighted KNN
ggplot(tri_roc, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path(size = 1.2) +
  geom_abline(lty = "dashed", color = "grey50") +
  labs(
    title = "ROC Curve: Triangular‐Weighted KNN",
    subtitle = paste0("k = ", tri_knn_best$neighbors, 
                      "; AUC = ", round(tri_knn_res %>% roc_auc(default, .pred_yes) %>% pull(.estimate), 3)),
    x     = "False Positive Rate",
    y     = "True Positive Rate",
    color = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")


# -----------------------------
# 15. Weighted KNN (inverse‐distance) -- corrected
# -----------------------------

weighted_knn_spec <- nearest_neighbor(
  mode        = "classification",
  neighbors   = tune(),
  dist_power  = 2,
  weight_func = "inv"        # ← inverse‐distance weighting
) %>%
  set_engine("kknn")          # no 'kernel' arg here

weighted_knn_wf <- workflow() %>%
  add_model(weighted_knn_spec) %>%
  add_recipe(knn_recipe)

weighted_knn_tune <- tune_grid(
  weighted_knn_wf,
  resamples = cv_folds,
  grid      = knn_grid,
  metrics   = metrics,
  control   = control_grid(save_pred = TRUE)
)

# name the metric argument explicitly!
weighted_knn_best <- select_best(weighted_knn_tune, metric = "roc_auc")

final_weighted_knn <- finalize_workflow(
  weighted_knn_wf,
  weighted_knn_best
) %>%
  fit(data = train_data)

# now predict on test set
weighted_res <- test_data %>%
  bind_cols(predict(final_weighted_knn, test_data, type = "prob")) %>%
  bind_cols(predict(final_weighted_knn, test_data, type = "class")) %>%
  rename(.pred_class_wknn = .pred_class)

# -----------------------------
# 16. Cost Calculation & Plotting (Corrected)
# -----------------------------

# 16.1 Compute costs for all models using the consistent sapply pattern
# Note: knn_costs and rf_costs were presumably calculated correctly earlier
#       using this same pattern. We are just correcting wknn_costs.

# Ensure thresholds and compute_cost are defined as before
thresholds <- seq(0, 1, by = 0.01)
cost_fp    <- 1
cost_fn    <- 5
compute_cost <- function(probs, actual, th) {
  pred <- if_else(probs >= th, "yes", "no")
  fp   <- sum(pred=="yes" & actual=="no")
  fn   <- sum(pred=="no"  & actual=="yes")
  cost_fp*fp + cost_fn*fn
}

# Recalculate knn_costs and rf_costs just to be sure (optional, if already done)
knn_costs <- sapply(thresholds, compute_cost,
                    probs = knn_res$.pred_yes,
                    actual = knn_res$default)
rf_costs  <- sapply(thresholds, compute_cost,
                    probs = rf_res$.pred_yes,
                    actual = rf_res$default)

# *** Corrected Calculation for Weighted KNN ***
wknn_costs <- sapply(thresholds, compute_cost,
                     probs = weighted_res$.pred_yes,
                     actual = weighted_res$default)

# Find optimal thresholds using the calculated costs
wknn_opt <- thresholds[which.min(wknn_costs)]
knn_opt  <- thresholds[which.min(knn_costs)] # Assuming this was done before
rf_opt   <- thresholds[which.min(rf_costs)]  # Assuming this was done before

# 16.2. Combine costs for all three models
library(tidyr)
cost_df <- tibble(
  threshold = thresholds,
  `KNN (unweighted)` = knn_costs,
  `KNN (weighted)`   = wknn_costs,  # Use the correctly calculated costs
  `Random Forest`    = rf_costs
) %>%
  pivot_longer(-threshold, names_to = "model", values_to = "total_cost")

# 16.3. Find optimal thresholds data frame for plotting
opt_df <- cost_df %>%
  group_by(model) %>%
  slice_min(total_cost, with_ties = FALSE) %>%
  ungroup()

# 16.4. Plot all three on one chart (using ggrepel and explicit ggplot2::margin)
library(ggplot2)
library(scales)
library(ggrepel)
library(dplyr) # Ensure dplyr is loaded for the pipe %>.%

# Ensure opt_df has the label formatted correctly
opt_df <- opt_df %>%
  mutate(label_text = paste0(percent(threshold, accuracy = 0.1), "\n(Cost: ", comma(total_cost, accuracy = 1), ")"))

ggplot(cost_df, aes(x = threshold, y = total_cost, color = model)) +
  geom_line(linewidth = 1.1) +
  geom_vline(data = opt_df, aes(xintercept = threshold, color = model),
             linetype = "dashed", linewidth = 0.7, alpha = 0.8) +
  geom_point(data = opt_df, aes(x = threshold, y = total_cost, color = model),
             size = 3.5) +
  ggrepel::geom_text_repel(
    data = opt_df,
    aes(x = threshold, y = total_cost, label = label_text),
    fontface = "bold",
    size = 3.5,
    nudge_y = 200,
    nudge_x = 0.02,
    segment.color = 'grey50',
    segment.size = 0.5,
    min.segment.length = 0.1,
    box.padding = 0.4,
    point.padding = 0.3,
    show.legend = FALSE
  ) +
  scale_color_manual(values = c(
    "KNN (unweighted)" = "#5F4B8B",
    "KNN (weighted)"   = "#009E73",
    "Random Forest"    = "#D55E00"
  )) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0.05, 0.1))) +
  scale_x_continuous(labels = percent) +
  labs(
    title    = "Total Prediction Cost vs. Probability Threshold",
    subtitle = paste("Cost = (False Positives ×", cost_fp, ") + (False Negatives ×", cost_fn, ")"),
    x        = "Probability Threshold (Predict “yes” if ≥ threshold)",
    y        = "Total Cost on Test Set",
    color    = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position  = "bottom",
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold"),
    # *** THIS LINE USES EXPLICIT NAMESPACE ***
    plot.subtitle = element_text(size = 11, margin = ggplot2::margin(b = 15, unit = "pt")),
    axis.title = element_text(face = "bold"),
    legend.title = element_text(face = "bold")
  )
# -----------------------------
# 17. Combined ROC Plot for All 3 Models
# -----------------------------
library(yardstick)
library(ggplot2)
library(dplyr)
library(stringr) # For str_wrap

# --- Ensure you have these result data frames from previous steps ---
# knn_res      <- test_data %>% bind_cols(...) # From section 11
# rf_res       <- test_data %>% bind_cols(...) # From section 11
# weighted_res <- test_data %>% bind_cols(...) # From section 15/16
# knn_best     <- select_best(knn_tune, metric = "roc_auc") # From section 9
# weighted_knn_best <- select_best(weighted_knn_tune, metric = "roc_auc") # From section 15
# rf_best      <- select_best(rf_tune, metric = "roc_auc") # From section 9
# -------------------------------------------------------------------

# ──────────────────────────────────────────────────────────────────────────────
# Combined ROC Plot for All 3 Models — Final Version
# ──────────────────────────────────────────────────────────────────────────────

library(yardstick)
library(ggplot2)
library(dplyr)
library(stringr)
# For non‐overlapping text labels:
# install.packages("ggrepel")
library(ggrepel)

# 1) Dynamic labels
knn_label           <- paste0("KNN (Unweighted, k=",    knn_best$neighbors,         ")")
weighted_knn_label  <- paste0("KNN (Weighted, k=",      weighted_knn_best$neighbors,")")
rf_label            <- "Random Forest"

# 2) ROC data
roc_data_all <- bind_rows(
  roc_curve(knn_res,      truth = default, .pred_yes) %>% mutate(model = knn_label),
  roc_curve(weighted_res, truth = default, .pred_yes) %>% mutate(model = weighted_knn_label),
  roc_curve(rf_res,       truth = default, .pred_yes) %>% mutate(model = rf_label)
)

# 3) AUC & annotation positions
auc_vals_all <- bind_rows(
  roc_auc(knn_res,      truth = default, .pred_yes) %>% mutate(model = knn_label),
  roc_auc(weighted_res, truth = default, .pred_yes) %>% mutate(model = weighted_knn_label),
  roc_auc(rf_res,       truth = default, .pred_yes) %>% mutate(model = rf_label)
) %>%
  select(model, .estimate) %>%
  mutate(
    x_pos = 0.55,
    y_pos = c(0.38, 0.28, 0.18),
    label = paste0(model, "\nAUC = ", round(.estimate, 3))
  )

# 4) Named colours
model_colors <- setNames(
  c("#5F4B8B", "#009E73", "#D55E00"),
  c(knn_label, weighted_knn_label, rf_label)
)

# 5) Plot
ggplot(roc_data_all, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path(linewidth = 1.4) +
  geom_abline(linetype = "dashed", color = "grey60") +
  
  # A) or B) for AUC labels:
  # A) geom_text(...)  
  # geom_text(data = auc_vals_all,
  #           aes(x = x_pos, y = y_pos, label = label, color = model),
  #           hjust = 0, vjust = 0, size = 4.2, fontface = "bold",
  #           show.legend = FALSE) +
  
  # B) with ggrepel (recommended)
  geom_text_repel(data = auc_vals_all,
                  aes(x = x_pos, y = y_pos, label = label, color = model),
                  min.segment.length = 0,
                  box.padding = 0.4,
                  size = 4,
                  fontface = "bold",
                  show.legend = FALSE) +
  
  scale_color_manual(values = model_colors) +
  labs(
    title    = "ROC Curves for Final Models",
    subtitle = str_wrap(
      "Unweighted KNN, Weighted KNN,Random Forest Classifiers",
      width = 60
    ),
    x     = "False Positive Rate (1 − Specificity)",
    y     = "True Positive Rate (Sensitivity)",
    color = "Model"
  ) +
  coord_fixed() +
  theme_minimal(base_size = 15) +
  theme(
    plot.title      = element_text(face = "bold", size = 18),
    plot.subtitle   = element_text(
      size   = 13,
      margin = ggplot2::margin(t = 0, r = 0, b = 10, l = 0, unit = "pt")
    ),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold"),
    legend.text      = element_text(size = 11),
    panel.grid.major = element_line(color = "grey90")
  )


# ───────────────────────────────────────────────────────────────────────────────
# 0. Libraries
# ───────────────────────────────────────────────────────────────────────────────
library(tidymodels)   # includes workflows, dplyr, ggplot2, etc.
library(ranger)       # for the ranger object
library(forcats)      # for fct_reorder()

# ───────────────────────────────────────────────────────────────────────────────
# 1. Extract the fitted ranger object
# ───────────────────────────────────────────────────────────────────────────────
# pull_workflow_fit() returns a "model_fit" with a `$fit` slot:
rf_fit_obj <- pull_workflow_fit(final_rf_fit)
# that `$fit` is the actual ranger object:
ranger_obj <- rf_fit_obj$fit

# ───────────────────────────────────────────────────────────────────────────────
# 2. Get the raw variable importances
# ───────────────────────────────────────────────────────────────────────────────
# ranger stores them in $variable.importance
imp_raw <- ranger_obj$variable.importance
# it's a named numeric vector:   names(imp_raw)  → original column names

# ───────────────────────────────────────────────────────────────────────────────
# 3. Create a lookup of pretty labels
# ───────────────────────────────────────────────────────────────────────────────
var_desc <- c(
  repayment_status_september_2005 = "Repayment Status (Sep 2005)",
  repayment_status_august_2005   = "Repayment Status (Aug 2005)",
  bill_statement_september_2005  = "Bill Statement (Sep 2005)",
  repayment_status_july_2005     = "Repayment Status (Jul 2005)",
  amount_paid_september_2005     = "Amount Paid (Sep 2005)",
  limit_bal                      = "Credit Limit",
  bill_statement_august_2005     = "Bill Statement (Aug 2005)",
  amount_paid_august_2005        = "Amount Paid (Aug 2005)",
  bill_statement_july_2005       = "Bill Statement (Jul 2005)",
  age                            = "Age",
  amount_paid_july_2005          = "Amount Paid (Jul 2005)",
  bill_statement_may_2005        = "Bill Statement (May 2005)",
  bill_statement_june_2005       = "Bill Statement (Jun 2005)",
  bill_statement_april_2005      = "Bill Statement (Apr 2005)",
  amount_paid_april_2005         = "Amount Paid (Apr 2005)",
  repayment_status_may_2005      = "Repayment Status (May 2005)",
  amount_paid_june_2005          = "Amount Paid (Jun 2005)",
  amount_paid_may_2005           = "Amount Paid (May 2005)",
  repayment_status_june_2005     = "Repayment Status (Jun 2005)",
  repayment_status_april_2005    = "Repayment Status (Apr 2005)"
)

# ───────────────────────────────────────────────────────────────────────────────
# 4. Build a small tibble, filter to top 20 & attach labels
# ───────────────────────────────────────────────────────────────────────────────
imp_df <- tibble(
  raw_name   = names(imp_raw),
  importance = as.numeric(imp_raw)
) %>%
  # keep only those you have descriptions for
  filter(raw_name %in% names(var_desc)) %>%
  mutate(
    label = var_desc[raw_name]
  ) %>%
  arrange(importance) %>%     # ascending
  slice_tail(n = 20)          # pick the top 20

# ───────────────────────────────────────────────────────────────────────────────
# 5. Plot
# ───────────────────────────────────────────────────────────────────────────────
ggplot(imp_df, aes(x = importance, y = fct_reorder(label, importance))) +
  geom_point(size = 3, color = "#2c7fb8") +
  labs(
    title    = "Top 20 Most Important Variables (Random Forest)",
    subtitle = "Impurity (mean decrease in Gini) from the ranger engine",
    x        = "Importance Score",
    y        = NULL
  ) +
  theme_minimal(base_size = 14)

library(ggplot2)
library(forcats)

p <- ggplot(imp_df, aes(x = importance, y = fct_reorder(label, importance))) +
  geom_point(size = 3, color = "#2c7fb8") +
  labs(
    title    = "Top 20 Most Important Variables\n(Random Forest Impurity)",
    subtitle = "Mean decrease in Gini, extracted from the ranger engine",
    x        = "Importance Score",
    y        = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.y   = element_text(
      size   = 10,
      margin = ggplot2::margin(r = 8)   # <- explicitly ggplot2::margin
    ),
    plot.title    = element_text(
      size       = 16,
      face       = "bold",
      hjust      = 0.5,
      lineheight = 1.2
    ),
    plot.subtitle = element_text(size = 12, hjust = 0.5),
    plot.margin   = ggplot2::margin(t = 10, r = 20, b = 10, l = 10)  # also explicit
  )

print(p)
ggsave("rf_importance.png", p, width = 9, height = 6)


