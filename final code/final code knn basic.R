# MA429 Final code compliation

# dataset link https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

rm(list = ls())
options(scipen = 999)

# setwd()

### Packages ###

library(tidyverse)
library(caret)
library(kknn)
library(class)

### Data Cleaning ### 

data = read.csv("dataset.csv") # dataset originally as a .xls file; using Excel, save as .csv file instead and then run code

data = data[,-1] # removing row ID

# for easier interpretability, re-ordering rows to chronological order (April to Sep instead of Sep to April) and then renaming each variable

data = data[ , c(1:5, 11:6, 17:12, 23:18, 24)]


colnames(data)[c(6:11)] = c("Repayment status April 2005", "Repayment status May 2005", "Repayment status June 2005", "Repayment status July 2005", "Repayment status August 2005", "Repayment status September 2005")

colnames(data)[c(12:17)] = c("Bill Statement April 2005", "Bill Statement May 2005", "Bill Statement June 2005", "Bill Statement July 2005", "Bill Statement August 2005", "Bill Statement September 2005")

colnames(data)[c(18:23)] = c("Amount Paid April 2005", "Amount Paid May 2005", "Amount Paid June 2005", "Amount Paid July 2005", "Amount Paid August 2005", "Amount Paid September 2005")

colnames(data)[24] = "Default"

# Some variable ranges were inaccurately provided by the UCI ML Repository description (see Kaggle form discussion for more details : https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/discussion/34608 )

# Hence, we provide the following appropriate transformations to the entire dataset 

data[, 3] <- ifelse(data[, 3] >= 4, 0, data[, 3])

summary(data)

# ensuring each variable is in the correct class

data[] = lapply(data, as.numeric)
data[,c(2,3,4,24)] = lapply(data[c(2,3,4,24)], as.factor)

### EDA ###

# Start by spliting the data into test and train so that no decisions made are based on test set

set.seed(10)
n <- nrow(data)
train_idx <- sample.int(n, size = 0.8*n) # 80% for training
train <- data[train_idx, ]
test  <- data[-train_idx, ]

# train_ohe = data_ohe[train_idx, ]
# test_ohe = data_ohe[-train_idx, ]




# visualising the training distributions of the repayment status variables

df_long <- train %>%
  pivot_longer(
    cols = starts_with("Repayment status"),
    names_to  = "month",
    values_to = "delay"
  )

ggplot(df_long, aes(x = delay)) +
  geom_histogram(binwidth=1, boundary=0.5, closed="right",
                 fill="steelblue", colour="white") +
  scale_x_continuous(breaks=sort(unique(df_long$delay))) +
  facet_wrap(~ month, ncol = 3) +
  labs(x="Repayment Delay (months) in Training set", y="Count") +
  theme_minimal()



# range is from -2 to 8 but number of instances between 3-8 for each variable small. AS this is an ordinal categorical variable we define a new category of 3 or more months delay; make this decision based on train data and apply to entire data 

data[, 6:11] <- lapply(data[, 6:11], function(x) 
  ifelse(x >= 4, 3, x)
)

# visualising training data new transformation  


df_long <- train %>%
  pivot_longer(
    cols = starts_with("Repayment status"),
    names_to  = "month",
    values_to = "delay"
  )

ggplot(df_long, aes(x = delay)) +
  geom_histogram(binwidth=1, boundary=0.5, closed="right",
                 fill="steelblue", colour="white") +
  scale_x_continuous(breaks=sort(unique(df_long$delay))) +
  facet_wrap(~ month, ncol = 3) +
  labs(x="Repayment Delay (months) in Training set", y="Count") +
  theme_minimal()


summary(data$`Bill Statement April 2005`)
summary(data$`Bill Statement May 2005`)
summary(data$`Bill Statement June 2005`)
summary(data$`Bill Statement July 2005`)
summary(data$`Bill Statement August 2005`)
summary(data$`Bill Statement September 2005`)



# visualising the training distributions of the  bill statement variables; boxplots comparing bill statements between months based on default

df_bs <- train %>%
  select(Default, starts_with("Bill statement")) %>%
  pivot_longer(
    cols      = -Default,
    names_to  = "month",
    values_to = "amount"
  ) %>%
  mutate(
    Default = factor(Default, levels = c("0","1"),
                     labels = c("No default","Default"))
  )

ggplot(df_bs, aes(x = Default, y = amount, fill = Default)) +
  geom_boxplot(na.rm = TRUE) +
  facet_wrap(~ month, ncol = 3) +
  labs(
    x     = "Default Status",
    y     = "Bill Statement Amount",
    title = "Bill Statement by Default Status, Apr–Sep 2005"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x     = element_text(angle = 45, hjust = 1)
  )

# visualising the training distributions of the  Amount repaid variables; boxplots comparing amount repaid between months based on default


df_bs <- train %>%
  select(Default, starts_with("Amount")) %>%
  pivot_longer(
    cols      = -Default,
    names_to  = "month",
    values_to = "amount"
  ) %>%
  mutate(
    Default = factor(Default, levels = c("0","1"),
                     labels = c("No default","Default"))
  )

ggplot(df_bs, aes(x = Default, y = log(amount), fill = Default)) +
  geom_boxplot(na.rm = TRUE) +
  facet_wrap(~ month, ncol = 3) +
  labs(
    x     = "Default Status",
    y     = "log(Amount Paid)",
    title = "Amount Paid by Default Status, Apr–Sep 2005"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x     = element_text(angle = 45, hjust = 1)
  )


summary(data$`Amount Paid April 2005`)
summary(data$`Amount Paid May 2005`)
summary(data$`Amount Paid June 2005`)
summary(data$`Amount Paid July 2005`)
summary(data$`Amount Paid August 2005`)
summary(data$`Amount Paid September 2005`)

# other numeric variables seem fine, no changes made
# only change made was to the ordinal categorical variable of Repayment delay of introducing a new category: 3 or more months

rm(df_bs)
rm(df_long)


# creating one hot encoded data set (to be used for knn)

fac_cols <- c(2,3,4)
fac_names <- names(data)[fac_cols]

# set up the dummy‐var transformer
dv <- dummyVars(~ ., data = data[fac_cols], fullRank = T)

# create the dummy data frame
dummies <- predict(dv, newdata = data)
dummies <- as.data.frame(dummies)

# combine with the non‐factor columns
data_ohe <- cbind(data[ , -fac_cols], dummies)

write.csv(data_ohe, "ohe28_cleaned_data.csv", row.names = FALSE) # note this will be used in next R file

summary(data)
summary(data_ohe)

# re-creating train and test data (same seed), after transformation applied to entire data based on training data EDA

set.seed(10)
n <- nrow(data)
train_idx <- sample.int(n, size = 0.8*n) # 80% for training
train <- data[train_idx, ]
test  <- data[-train_idx, ]

train_ohe = data_ohe[train_idx, ]
test_ohe = data_ohe[-train_idx, ]

# cleaning environment 
rm(dummies)
rm(dv)
rm(fac_cols)
rm(fac_names)
rm(n)
rm(train_idx)

# standardising based on training data before knn

sdevs <- sapply(train_ohe[,c(1:20)], sd)
means <- colMeans(train_ohe[,c(1:20)])
train_ohe[,c(1:20)] <- scale(train_ohe[,c(1:20)], center=means, scale=sdevs)
test_ohe[,c(1:20)] <- scale(test_ohe[,c(1:20)], center=means, scale=sdevs)

summary(train_ohe)
summary(test_ohe)


# cross validation split for knn parameter tuning

# define utility function 

# Utility function
utility <- function(true, pred) {
  if      (true=="1" && pred=="1")  0
  else if (true=="0" && pred=="1") -1
  else if (true=="0" && pred=="0")  0
  else if (true=="1" && pred=="0") -5
  else NA
}


# set up 5‐fold CV
tr_ctrl <- trainControl(method = "cv", number = 5, savePredictions = "all")

# grid of k values
grid <- expand.grid(k = seq(1, 451, 50))

# train
knn_fit <- train(
  x          = train_ohe[, c(1:20,22:28)],
  y          = as.factor(train_ohe[, 21]),
  method     = "knn",
  tuneGrid   = grid,
  trControl  = tr_ctrl
)

# finding best k
knn_fit # can see highest accuracy

res <- knn_fit$pred

res$score <- mapply(utility,
                    true = res$obs,
                    pred = res$pred)

# 2. for each k, build a confusion matrix and extract Sensitivity
sens_by_k <- sapply(unique(res$k), function(K) {
  sub <- subset(res, k == K)
  cm  <- confusionMatrix(
    data     = factor(sub$pred, levels = levels(sub$obs)),
    reference= factor(sub$obs,  levels = levels(sub$obs)),
    positive = "1"
  )
  cm$byClass["Sensitivity"]
})

agg_score   <- aggregate(score ~ k, data = res, FUN = sum)
score_by_k  <- agg_score$score

# view as a data.frame
data.frame(
  k        = seq(1,451,50),
  Accuracy = knn_fit$results$Accuracy,
  Recall   = as.numeric(sens_by_k),
  Cost    = score_by_k 
  
)

# optimal K for accuracy, recall and cost matrix lies between 1 and 101, so re-fit knn for subset range to find optimal k

# grid <- expand.grid(k = seq(1, 101, 2)) , ran this once but took long, found k that best minimises cost to be k = 21, so run on reduced no. of params just to show same result

grid = expand.grid(k = seq(1, 35, 2))

# train
knn_fit <- train(
  x          = train_ohe[, c(1:20,22:28)],
  y          = as.factor(train_ohe[, 21]),
  method     = "knn",
  tuneGrid   = grid,
  trControl  = tr_ctrl
)

# finding best k
knn_fit # can see highest accuracy

res <- knn_fit$pred

res$score <- mapply(utility,
                    true = res$obs,
                    pred = res$pred)

# 2. for each k, build a confusion matrix and extract Sensitivity
sens_by_k <- sapply(unique(res$k), function(K) {
  sub <- subset(res, k == K)
  cm  <- confusionMatrix(
    data     = factor(sub$pred, levels = levels(sub$obs)),
    reference= factor(sub$obs,  levels = levels(sub$obs)),
    positive = "1"
  )
  cm$byClass["Sensitivity"]
})

agg_score   <- aggregate(score ~ k, data = res, FUN = sum)
score_by_k  <- agg_score$score

# view as a data.frame
results = data.frame(k = seq(1,35,2), Accuracy = knn_fit$results$Accuracy, Recall   = as.numeric(sens_by_k), Cost    = score_by_k)

results[which.max(results$Accuracy), ] # note that for FN cost 10 times larger than FP cost, accuracy maximised at 29
results[which.max(results$Recall), ] # note that for FN cost 10 times larger than FP cost, recall maximised at 1
results[which.max(results$Cost), ] # note that for FN cost 10 times larger than FP cost, utility maximised at 1


### OPTIMAL K for minimising cost is 21, for K = 21, now consider variable subset selection

rm(means, score_by_k, sdevs, sens_by_k, res, agg_score)


############# defining fucntion to add variables at each stage based on the improvement to utility score

## NOTE: next section will take long to run as using cross-validation for forward variable selection (at each stage considers all remaining variables that could be included and aggregates 5 validation utilities for each variable e.g. from first step considers 27 x 5 different models)

# Utility function (vectorized)
utility <- function(true, pred) {
  t <- as.character(true)
  p <- as.character(pred)
  u <- ifelse(
    t == "1" & p == "1",  0,
    ifelse(
      t == "0" & p == "1", -1,
      ifelse(
        t == "0" & p == "0",  0,
        ifelse(
          t == "1" & p == "0", -5,
          NA
        )
      )
    )
  )
  return(u)
}

fixed_k   <- 21
all_feats <- setdiff(seq_len(ncol(train_ohe)), 21)
selected  <- integer(0)
best_util <- -Inf




# using 5 fold cross validation for each utility estimate for variable subset
set.seed(2025)
folds <- sample(rep(seq_len(5), length.out = nrow(train_ohe)))

repeat {
  remaining <- setdiff(all_feats, selected)
  utils     <- sapply(remaining, function(f) {
    feats  <- c(selected, f)
    fold_u <- numeric(5)
    for (i in seq_len(5)) {
      tr_idx <- which(folds != i)
      va_idx <- which(folds == i)
      
      # feature matrices
      train_mat <- as.matrix(train_ohe[tr_idx, feats, drop = FALSE])
      valid_mat <- as.matrix(train_ohe[va_idx, feats, drop = FALSE])
      
      # to stop large number of ties, add some small noise to each
      eps       <- 1e-8
      train_mat <- train_mat + matrix(rnorm(length(train_mat), 0, eps),
                                      nrow = nrow(train_mat))
      valid_mat <- valid_mat + matrix(rnorm(length(valid_mat), 0, eps),
                                      nrow = nrow(valid_mat))
      
      # k-NN with tie‐breaking
      pred <- knn(
        train   = train_mat,
        test    = valid_mat,
        cl      = train_ohe[tr_idx, 21],
        k       = fixed_k,
        use.all = TRUE
      )
      
      fold_u[i] <- sum(
        utility(true = train_ohe[va_idx, 21],
                pred = pred)
      )
    }
    mean(fold_u)
  })
  
  best_idx <- which.max(utils)
  new_util <- utils[best_idx]
  
  if (new_util > best_util) {
    selected  <- c(selected, remaining[best_idx])
    best_util <- new_util
  } else {
    break
  }
}

# Final selected features and best utility
selected 
best_util

colnames(train_ohe[,selected])



# algorithm taking greedy approach but seems to like a range of repayment status and bill statement; from a different seed (no. 10), the following were selected: 3, 7, 8, 11, 14 

# Train on full set and compute confusion matrix

final_train_knn <- knn(train = train_ohe[, selected], test = test_ohe[,selected], k = 21, use.all = T, cl = train_ohe[,21] )


test_cf_knn = confusionMatrix(
  data      = factor(final_train_knn, levels = levels(train_ohe[,21])),
  reference = factor(test_ohe[,21], levels = levels(train_ohe[,21])),
  positive  = "1"
)

test_cf_knn # final result from simple knn with hyper parameter tuning

best_util/4800 # average cost of for each validation sample, train sample is 24000 so each validation fold as 4800 obs 
(test_cf_knn$table[1,2] * -5 + test_cf_knn$table[2,1] * -1)/6000 # average cost for each test instance

###############



# Grid of thresholds to try
thresh_grid <- seq(1/21, 0.5, by = 1/21)

# Function to compute mean utility at a given threshold
eval_thresh <- function(thresh) {
  fold_utils <- numeric(5)
  for (i in seq_len(5)) {
    # split
    tr_idx <- which(folds != i); va_idx <- which(folds == i)
    tr_mat <- as.matrix(train_ohe[tr_idx, selected, drop=FALSE])
    va_mat <- as.matrix(train_ohe[va_idx, selected, drop=FALSE])
    # add tiny noise to break ties
    eps <- 1e-8
    tr_mat <- tr_mat + matrix(rnorm(length(tr_mat),0,eps), nrow=nrow(tr_mat))
    va_mat <- va_mat + matrix(rnorm(length(va_mat),0,eps), nrow=nrow(va_mat))
    # kNN with probabilities
    preds <- knn(train=tr_mat, test=va_mat,
                 cl=train_ohe[tr_idx,21],
                 k=fixed_k, use.all=TRUE, prob=TRUE)
    # extract positive‐class vote proportion
    prob_win <- attr(preds, "prob")
    p_pos <- ifelse(preds == "1", prob_win, 1 - prob_win)
    # apply threshold
    final_pred <- factor(ifelse(p_pos >= thresh, "1", "0"),
                         levels=levels(train_ohe[,21]))
    # utility sum
    fold_utils[i] <- sum(utility(true = train_ohe[va_idx,21],
                                 pred = final_pred))
  }
  mean(fold_utils)
}

# evaluate all thresholds
mean_utils <- sapply(thresh_grid, eval_thresh)

# pick best
best_thresh <- thresh_grid[which.max(mean_utils)]
best_thresh
mean_utils[which.max(mean_utils)]

# Final evaluation on test set
# get probabilities on test
final_knn <- knn(train = as.matrix(train_ohe[,selected]),
                 test  = as.matrix(test_ohe[,selected]),
                 cl    = train_ohe[,21],
                 k     = fixed_k, use.all=TRUE, prob=TRUE)
prob_win_test <- attr(final_knn, "prob")
p_pos_test    <- ifelse(final_knn == "1", prob_win_test, 1 - prob_win_test)
pred_test     <- factor(ifelse(p_pos_test >= best_thresh, "1", "0"),
                        levels=levels(train_ohe[,21]))
# confusion & utility
test_cm <- confusionMatrix(data=pred_test,
                           reference=factor(test_ohe[,21], levels=levels(train_ohe[,21])),
                           positive="1")
test_cm
test_utility <- sum(utility(true = test_ohe[,21], pred = pred_test))
avg_test_cost <- test_utility / nrow(test_ohe)
avg_test_cost






##########

      
# clean envi
rm(grid, knn_fit, results, tr_ctrl, all_feats, best_idx, best_util, final_train_knn, fixed_k, new_util, remaining, selected, utils )

######### Now consider if PCA improves performance

pca_train <- prcomp(train_ohe[,-21], center = T, scale. = T)
pca_test <- predict(pca_train, newdata=test_ohe[,-21])


summary(pca_train)

pca_train_df <- as.data.frame(pca_train$x)

pca_test_df <- as.data.frame(pca_test)

pca_train_df[,28] <-train_ohe[,21]
pca_test_df[,28] <-test_ohe[,21]

##### standardising data for knn

sdevs <- sapply(pca_train_df[,c(1:27)], sd)
means <- colMeans(pca_train_df[,c(1:27)])
pca_train_df[,c(1:27)] <- scale(pca_train_df[,c(1:27)], center=means, scale=sdevs)
pca_test_df[,c(1:27)] <- scale(pca_test_df[,c(1:27)], center=means, scale=sdevs)





# Utility function
utility <- function(true, pred) {
  if      (true=="1" && pred=="1")  0
  else if (true=="0" && pred=="1") -1
  else if (true=="0" && pred=="0")  0
  else if (true=="1" && pred=="0") -5
  else NA
}


# set up 5‐fold CV
tr_ctrl <- trainControl(method = "cv", number = 5, savePredictions = "all")

# grid of k values
grid <- expand.grid(k = seq(1, 451, 50))

# train
pca_knn_fit <- train(
  x          = pca_train_df[, -28],
  y          = as.factor(pca_train_df[, 28]),
  method     = "knn",
  tuneGrid   = grid,
  trControl  = tr_ctrl
)

# finding best k
pca_knn_fit # can see highest accuracy

res <- pca_knn_fit$pred

res$score <- mapply(utility,
                    true = res$obs,
                    pred = res$pred)

# 2. for each k, build a confusion matrix and extract Sensitivity
sens_by_k <- sapply(unique(res$k), function(K) {
  sub <- subset(res, k == K)
  cm  <- confusionMatrix(
    data     = factor(sub$pred, levels = levels(sub$obs)),
    reference= factor(sub$obs,  levels = levels(sub$obs)),
    positive = "1"
  )
  cm$byClass["Sensitivity"]
})

agg_score   <- aggregate(score ~ k, data = res, FUN = sum)
score_by_k  <- agg_score$score

# view as a data.frame
data.frame(
  k        = seq(1,451,50),
  Accuracy = pca_knn_fit$results$Accuracy,
  Recall   = as.numeric(sens_by_k),
  Cost    = score_by_k 
  
)


# optimal K for accuracy, recall and cost matrix lies between 1 and 51, so re-fit knn for subset range to find optimal k
# NOTE will take time to run, optimal k found was : 

grid = expand.grid(k = seq(1, 51, 2))

# train
pca_knn_fit <- train(
  x          = pca_train_df[, -28],
  y          = as.factor(pca_train_df[, 28]),
  method     = "knn",
  tuneGrid   = grid,
  trControl  = tr_ctrl
)

# finding best k
pca_knn_fit # can see highest accuracy

res <- pca_knn_fit$pred

res$score <- mapply(utility,
                    true = res$obs,
                    pred = res$pred)

# 2. for each k, build a confusion matrix and extract Sensitivity
sens_by_k <- sapply(unique(res$k), function(K) {
  sub <- subset(res, k == K)
  cm  <- confusionMatrix(
    data     = factor(sub$pred, levels = levels(sub$obs)),
    reference= factor(sub$obs,  levels = levels(sub$obs)),
    positive = "1"
  )
  cm$byClass["Sensitivity"]
})

agg_score   <- aggregate(score ~ k, data = res, FUN = sum)
score_by_k  <- agg_score$score


# view as a data.frame
results = data.frame(k = seq(1,51,2), Accuracy = pca_knn_fit$results$Accuracy, Recall   = as.numeric(sens_by_k), Cost    = score_by_k)

results[which.max(results$Accuracy), ] 
results[which.max(results$Recall), ] 
results[which.max(results$Cost), ] 


#############


### OPTIMAL K for minimising cost is 7, for K = 7, now consider variable subset selection

rm(score_by_k, sens_by_k, res, agg_score)


############# defining fucntion to add variables at each stage based on the improvement to utility score

## 

# Utility function (vectorized)
utility <- function(true, pred) {
  t <- as.character(true)
  p <- as.character(pred)
  u <- ifelse(
    t == "1" & p == "1",  0,
    ifelse(
      t == "0" & p == "1", -1,
      ifelse(
        t == "0" & p == "0",  0,
        ifelse(
          t == "1" & p == "0", -5,
          NA
        )
      )
    )
  )
  return(u)
}

fixed_k   <- 7
all_feats <- setdiff(seq_len(ncol(pca_train_df)), 28)
selected  <- integer(0)
best_util <- -Inf




# using 5 fold cross validation for each utility estimate for variable subset
set.seed(2025)
# using same folds from earlier 

repeat {
  remaining <- setdiff(all_feats, selected)
  utils     <- sapply(remaining, function(f) {
    feats  <- c(selected, f)
    fold_u <- numeric(5)
    for (i in seq_len(5)) {
      tr_idx <- which(folds != i)
      va_idx <- which(folds == i)
      
      # feature matrices
      train_mat <- as.matrix(pca_train_df[tr_idx, feats, drop = FALSE])
      valid_mat <- as.matrix(pca_train_df[va_idx, feats, drop = FALSE])
      
      # to stop large number of ties, add some small noise to each
      eps       <- 1e-8
      train_mat <- train_mat + matrix(rnorm(length(train_mat), 0, eps),
                                      nrow = nrow(train_mat))
      valid_mat <- valid_mat + matrix(rnorm(length(valid_mat), 0, eps),
                                      nrow = nrow(valid_mat))
      
      # k-NN with tie‐breaking
      pred <- knn(
        train   = train_mat,
        test    = valid_mat,
        cl      = pca_train_df[tr_idx, 28],
        k       = fixed_k,
        use.all = TRUE
      )
      
      fold_u[i] <- sum(
        utility(true = pca_train_df[va_idx, 28],
                pred = pred)
      )
    }
    mean(fold_u)
  })
  
  best_idx <- which.max(utils)
  new_util <- utils[best_idx]
  
  if (new_util > best_util) {
    selected  <- c(selected, remaining[best_idx])
    best_util <- new_util
  } else {
    break
  }
}

# Final selected features and best utility
selected # ran this for a different seed as well (no. 10) and got the following selected variable index  8  7 19  5  6 22  1 20 (so very similar variable selection)

best_util


# algorithm taking greedy approach but seems to like a range of repayment status and bill statement; combining variable selection from both seeds, the following are shared in common: 3, 7, 8, 11, 14 and will be used to fit the knn model   

# Train on full set and compute confusion matrix

final_pca_knn <- knn(train = pca_train_df[, selected], test = pca_test_df[,selected], k = 5, use.all = T, cl = pca_train_df[,28] )

test_cf_pca_knn = confusionMatrix(
  data      = factor(final_pca_knn, levels = levels(pca_train_df[,28])),
  reference = factor(pca_test_df[,28], levels = levels(pca_train_df[,28])),
  positive  = "1"
)

test_cf_pca_knn # final result from pca knn with hyper parameter tuning

best_util/4800 # average cost of for each validation sample, train sample is 24000 so each validation fold as 4800 obs 
(test_cf_pca_knn$table[1,2] * -5 + test_cf_pca_knn$table[2,1] * -1)/6000 # average cost for each test instance


########



# threshold candidates: proportion of the k=5 neighbours
thresh_grid <- seq(1/5, 0.5, by = 1/5)

# evaluate a single threshold by 5-fold CV on the PCA training data
eval_thresh <- function(thresh) {
  fold_utils <- numeric(5)
  for (i in seq_len(5)) {
    tr_idx <- which(folds != i); va_idx <- which(folds == i)
    tr_mat <- as.matrix(pca_train_df[tr_idx, selected, drop=FALSE])
    va_mat <- as.matrix(pca_train_df[va_idx, selected, drop=FALSE])
    # break ties
    eps <- 1e-8
    tr_mat <- tr_mat + matrix(rnorm(length(tr_mat),0,eps), nrow=nrow(tr_mat))
    va_mat <- va_mat + matrix(rnorm(length(va_mat),0,eps), nrow=nrow(va_mat))
    # kNN with prob=TRUE
    preds <- knn(
      train   = tr_mat,
      test    = va_mat,
      cl      = factor(pca_train_df[tr_idx, 28]),  # ← use column 28
      k       = fixed_k,
      use.all = TRUE,
      prob    = TRUE
    )
    # proportion of “1” votes
    prob_win <- attr(preds, "prob")
    p_pos    <- ifelse(preds == "1", prob_win, 1 - prob_win)
    # threshold to final prediction
    final_pred <- factor(
      ifelse(p_pos >= thresh, "1", "0"),
      levels = c("0","1")
    )
    # sum utility on this fold
    fold_utils[i] <- sum(
      utility(
        true = as.character(pca_train_df[va_idx, 28]),  # ← col 28
        pred = as.character(final_pred)
      )
    )
  }
  mean(fold_utils)
}

# run over grid
mean_utils <- sapply(thresh_grid, eval_thresh)

# pick best
best_thresh <- thresh_grid[which.max(mean_utils)]
cat("best threshold =", best_thresh, "\n",
    "mean utility =", max(mean_utils), "\n\n")

# now apply to the PCA test set
final_knn <- knn(
  train   = as.matrix(pca_train_df[, selected]),
  test    = as.matrix(pca_test_df[, selected]),
  cl      = factor(pca_train_df[, 28]),  # ← col 28
  k       = fixed_k,
  use.all = TRUE,
  prob    = TRUE
)
prob_win_test <- attr(final_knn, "prob")
p_pos_test    <- ifelse(final_knn == "1", prob_win_test, 1-prob_win_test)
pred_test     <- factor(
  ifelse(p_pos_test >= best_thresh, "1", "0"),
  levels = c("0","1")
)

# confusion & cost
test_cm       <- confusionMatrix(pred_test,
                                 reference = factor(pca_test_df[,28], levels=c("0","1")),
                                 positive  = "1")
test_utility  <- sum(utility(true = as.character(pca_test_df[,28]),
                             pred = as.character(pred_test)))
avg_test_cost <- test_utility / nrow(pca_test_df)

max(mean_utils)/4800

avg_test_cost














### does the evaluation stand up to evaluation? a first principle approach to the evaluation of classifiers

rm(list = ls())


















