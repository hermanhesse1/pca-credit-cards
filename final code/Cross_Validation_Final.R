##### Deep Learning - Feedforward Neural Networks #####

### Good Approach: Try to get a better recall by bootstrapping

#rm(list=ls())

## Libraries and Seed

library(torch)
library(luz) # High-level interface for torch
library(dplyr)
torch_manual_seed(10)

## Preliminary Data Transformations

data = read.csv('cleaned_data.csv')

n <- nrow(data)

summary(data)

variable_names <- c("SEX", "EDUCATION", "MARRIAGE", "Default")

for (i in variable_names) {
  data[[i]] <- as.factor(data[[i]])
}

summary(data)

#levels(data$SEX)
#levels(data$EDUCATION)
#levels(data$MARRIAGE)
#levels(data$Default)

y = data$Default

# The following command automatically applies 1-hot encoding, and also splits
# out the features and the target. It also returns a matrix rather than a dataframe
# for the input variables. This is needed for torch, which does not work directly
# with R dataframes.

x = model.matrix(Default ~ . -1, data = data)
p = ncol(x)

# Put it again as dataframe, just wanted the 1-hot encoding for now

x <- as.data.frame(x)
x$Default <- y

# We take out one of the sex variables since it does not give us any extra information (even though it does not matter in this case)

x = x[,-3]
summary(x)

# Training/test split

set.seed(10)
ntest <- trunc(n / 5)
testid <- sample(1:n, ntest)
#test = x[testid,] - We do not use the test set here
train = x[-testid,]

# Standardise the training data and then the test data with the same transforation

store1 = train$Default # Store temporarily
#store2 = test$Default # Store temporarily - We do not use the test set here

sdevs = sapply(train[,-28], sd)
mn = colMeans(train[,-28])

scaled_mat <- scale(train[,-28], center = mn, scale = sdevs)
#scaled_mat_test = scale(test[,-28], center = mn, scale = sdevs) - We do not use the test set here

# Turn back to dataframe

train <- as.data.frame(scaled_mat)
#test <- as.data.frame(scaled_mat_test) - We do not use the test set here

# Add the targets back

train$Default <- store1
#test$Default <- store2 - We do not use the test set here

## Building the Networks

# 1

# Initialize the Network

modnn_1 <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(in_features = 27, out_features = 128)
    self$linear2 <- nn_linear(in_features = 128, out_features = 256)
    self$linear3 <- nn_linear(in_features = 256, out_features = 512)
    self$linear4 <- nn_linear(in_features = 512, out_features = 2)
    self$activation <- nn_relu()
    self$dropout1 <- nn_dropout(0.35)
    self$dropout2 <- nn_dropout(0.35)
    self$dropout3 <- nn_dropout(0.4)
  },
  forward = function(x) {
    x %>%
      self$linear1() %>% self$activation() %>% self$dropout1() %>%
      self$linear2() %>% self$activation() %>% self$dropout2() %>%
      self$linear3() %>% self$activation() %>% self$dropout3() %>%
      self$linear4()
  }
)

# Specify the loss and determine the optimizer

modelnn_1 <- modnn_1 %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = function(params) optim_rmsprop(params, lr = 0.001),
    metrics = list(luz_metric_accuracy())
  )

# 2

# Initialize the Network

modnn_2 <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(in_features = 27, out_features = 128)
    self$linear2 <- nn_linear(in_features = 128, out_features = 256)
    self$linear3 <- nn_linear(in_features = 256, out_features = 512)
    self$linear4 <- nn_linear(in_features = 512, out_features = 1024)
    self$linear5 <- nn_linear(in_features = 1024, out_features = 2)
    self$activation <- nn_relu()
    self$dropout1 <- nn_dropout(0.4)
    self$dropout2 <- nn_dropout(0.4)
    self$dropout3 <- nn_dropout(0.5)
    self$dropout4 <- nn_dropout(0.5)
  },
  forward = function(x) {
    x %>%
      self$linear1() %>% self$activation() %>% self$dropout1() %>%
      self$linear2() %>% self$activation() %>% self$dropout2() %>%
      self$linear3() %>% self$activation() %>% self$dropout3() %>%
      self$linear4() %>% self$activation() %>% self$dropout4() %>%
      self$linear5()
  }
)

# Specify the loss and determine the optimizer

modelnn_2 <- modnn_2 %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = function(params) optim_rmsprop(params, lr = 0.001),
    metrics = list(luz_metric_accuracy())
  )

# 3

# Initialize the Network

modnn_3 <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(in_features = 27, out_features = 50)
    self$linear2 <- nn_linear(in_features = 50, out_features = 100)
    self$linear3 <- nn_linear(in_features = 100, out_features = 50)
    self$linear4 <- nn_linear(in_features = 50, out_features = 2)
    self$activation <- nn_relu()
    self$dropout1 <- nn_dropout(0.4)
    self$dropout2 <- nn_dropout(0.4)
    self$dropout3 <- nn_dropout(0.4)
  },
  forward = function(x) {
    x %>%
      self$linear1() %>% self$activation() %>% self$dropout1() %>%
      self$linear2() %>% self$activation() %>% self$dropout2() %>%
      self$linear3() %>% self$activation() %>% self$dropout3() %>%
      self$linear4()
  }
)

# Specify the loss and determine the optimizer

modelnn_3 <- modnn_3 %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = function(params) optim_rmsprop(params, lr = 0.01),
    metrics = list(luz_metric_accuracy())
  )

# 4

# Initialize the Network

modnn_4 <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(in_features = 27, out_features = 32)
    self$linear2 <- nn_linear(in_features = 32, out_features = 64)
    self$linear3 <- nn_linear(in_features = 64, out_features = 108)
    self$linear4 <- nn_linear(in_features = 108, out_features = 2)
    self$activation <- nn_relu()
    self$dropout1 <- nn_dropout(0.4)
    self$dropout2 <- nn_dropout(0.4)
    self$dropout3 <- nn_dropout(0.4)
  },
  forward = function(x) {
    x %>%
      self$linear1() %>% self$activation() %>% self$dropout1() %>%
      self$linear2() %>% self$activation() %>% self$dropout2() %>%
      self$linear3() %>% self$activation() %>% self$dropout3() %>%
      self$linear4()
  }
)

# Specify the loss and determine the optimizer

modelnn_4 <- modnn_4 %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = function(params) optim_rmsprop(params, lr = 0.01),
    metrics = list(luz_metric_accuracy())
  )

# 5

# Initialize the Network

modnn_5 <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(in_features = 27, out_features = 32)
    self$linear2 <- nn_linear(in_features = 32, out_features = 64)
    self$linear3 <- nn_linear(in_features = 64, out_features = 108)
    self$linear4 <- nn_linear(in_features = 108, out_features = 216)
    self$linear5 <- nn_linear(in_features = 216, out_features = 2)
    self$activation <- nn_relu()
    self$dropout1 <- nn_dropout(0.35)
    self$dropout2 <- nn_dropout(0.4)
    self$dropout3 <- nn_dropout(0.4)
    self$dropout4 <- nn_dropout(0.4)
  },
  forward = function(x) {
    x %>%
      self$linear1() %>% self$activation() %>% self$dropout1() %>%
      self$linear2() %>% self$activation() %>% self$dropout2() %>%
      self$linear3() %>% self$activation() %>% self$dropout3() %>%
      self$linear4() %>% self$activation() %>% self$dropout4() %>%
      self$linear5()
  }
)

# Specify the loss and determine the optimizer

modelnn_5 <- modnn_5 %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = function(params) optim_rmsprop(params, lr = 0.001),
    metrics = list(luz_metric_accuracy())
  )

# 6

# Initialize the Network

modnn_6 <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(in_features = 27, out_features = 128)
    self$linear2 <- nn_linear(in_features = 128, out_features = 256)
    self$linear3 <- nn_linear(in_features = 256, out_features = 512)
    self$linear4 <- nn_linear(in_features = 512, out_features = 2)
    self$activation <- nn_softsign()
    self$dropout1 <- nn_dropout(0.35)
    self$dropout2 <- nn_dropout(0.35)
    self$dropout3 <- nn_dropout(0.4)
  },
  forward = function(x) {
    x %>%
      self$linear1() %>% self$activation() %>% self$dropout1() %>%
      self$linear2() %>% self$activation() %>% self$dropout2() %>%
      self$linear3() %>% self$activation() %>% self$dropout3() %>%
      self$linear4()
  }
)

# Specify the loss and determine the optimizer

modelnn_6 <- modnn_6 %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = function(params) optim_rmsprop(params, lr = 0.001),
    metrics = list(luz_metric_accuracy())
  )

# 7

# Initialize the Network

modnn_7 <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(in_features = 27, out_features = 128)
    self$linear2 <- nn_linear(in_features = 128, out_features = 256)
    self$linear3 <- nn_linear(in_features = 256, out_features = 512)
    self$linear4 <- nn_linear(in_features = 512, out_features = 1024)
    self$linear5 <- nn_linear(in_features = 1024, out_features = 2)
    self$activation <- nn_tanh()
    self$dropout1 <- nn_dropout(0.4)
    self$dropout2 <- nn_dropout(0.4)
    self$dropout3 <- nn_dropout(0.5)
    self$dropout4 <- nn_dropout(0.5)
  },
  forward = function(x) {
    x %>%
      self$linear1() %>% self$activation() %>% self$dropout1() %>%
      self$linear2() %>% self$activation() %>% self$dropout2() %>%
      self$linear3() %>% self$activation() %>% self$dropout3() %>%
      self$linear4() %>% self$activation() %>% self$dropout4() %>%
      self$linear5()
  }
)

# Specify the loss and determine the optimizer

modelnn_7 <- modnn_7 %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = function(params) optim_rmsprop(params, lr = 0.001),
    metrics = list(luz_metric_accuracy())
  )

# 8

# Initialize the Network

modnn_8 <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(in_features = 27, out_features = 50)
    self$linear2 <- nn_linear(in_features = 50, out_features = 100)
    self$linear3 <- nn_linear(in_features = 100, out_features = 50)
    self$linear4 <- nn_linear(in_features = 50, out_features = 2)
    self$activation <- nn_softplus()
    self$dropout1 <- nn_dropout(0.4)
    self$dropout2 <- nn_dropout(0.4)
    self$dropout3 <- nn_dropout(0.4)
  },
  forward = function(x) {
    x %>%
      self$linear1() %>% self$activation() %>% self$dropout1() %>%
      self$linear2() %>% self$activation() %>% self$dropout2() %>%
      self$linear3() %>% self$activation() %>% self$dropout3() %>%
      self$linear4()
  }
)

# Specify the loss and determine the optimizer

modelnn_8 <- modnn_8 %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = function(params) optim_rmsprop(params, lr = 0.01),
    metrics = list(luz_metric_accuracy())
  )

# 9

# Initialize the Network

modnn_9 <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(in_features = 27, out_features = 32)
    self$linear2 <- nn_linear(in_features = 32, out_features = 64)
    self$linear3 <- nn_linear(in_features = 64, out_features = 108)
    self$linear4 <- nn_linear(in_features = 108, out_features = 2)
    self$activation <- nn_tanh()
    self$dropout1 <- nn_dropout(0.4)
    self$dropout2 <- nn_dropout(0.4)
    self$dropout3 <- nn_dropout(0.4)
  },
  forward = function(x) {
    x %>%
      self$linear1() %>% self$activation() %>% self$dropout1() %>%
      self$linear2() %>% self$activation() %>% self$dropout2() %>%
      self$linear3() %>% self$activation() %>% self$dropout3() %>%
      self$linear4()
  }
)

# Specify the loss and determine the optimizer

modelnn_9 <- modnn_9 %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = function(params) optim_rmsprop(params, lr = 0.01),
    metrics = list(luz_metric_accuracy())
  )

# 10

# Initialize the Network

modnn_10 <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(in_features = 27, out_features = 32)
    self$linear2 <- nn_linear(in_features = 32, out_features = 64)
    self$linear3 <- nn_linear(in_features = 64, out_features = 108)
    self$linear4 <- nn_linear(in_features = 108, out_features = 216)
    self$linear5 <- nn_linear(in_features = 216, out_features = 2)
    self$activation <- nn_tanh()
    self$dropout1 <- nn_dropout(0.35)
    self$dropout2 <- nn_dropout(0.4)
    self$dropout3 <- nn_dropout(0.4)
    self$dropout4 <- nn_dropout(0.4)
  },
  forward = function(x) {
    x %>%
      self$linear1() %>% self$activation() %>% self$dropout1() %>%
      self$linear2() %>% self$activation() %>% self$dropout2() %>%
      self$linear3() %>% self$activation() %>% self$dropout3() %>%
      self$linear4() %>% self$activation() %>% self$dropout4() %>%
      self$linear5()
  }
)

# Specify the loss and determine the optimizer

modelnn_10 <- modnn_10 %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = function(params) optim_rmsprop(params, lr = 0.001),
    metrics = list(luz_metric_accuracy())
  )

## Train the Model (Bootstrap - CV)

# Do the folds for CV (with Bootstrap)

train <- train[sample(nrow(train)),]  # Shuffle training predictors

# Assign probabilities

train = as.data.frame(train)

# First we do some R trick to not get a factor
train$Default <- as.integer(as.character(train$Default))

# Then we assign weights in one line
train$w_inst <- ifelse(train$Default == 1, 5, 1) # To get a 50/50 split each time

# And we normalize to sum to 1
train$w_inst <- train$w_inst / sum(train$w_inst)

train$cumulative_value <- cumsum(train$w_inst) # Find cumulative value to partition into folds based on weights

train$group <- cut(train$cumulative_value, 
                   breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1), 
                   labels = 1:5,
                   right = TRUE, include.lowest = TRUE)

train$group <- as.integer(as.character(train$group)) # Just to make sure it is numeric

k_values <- seq(1, 10, by = 1)  # 10 architectures of Neural Networks to be tested

# A list where each element is length-5 numeric (one entry per fold)
all_recalls <- vector("list", length(k_values))
names(all_recalls) <- paste0("k", k_values)

for (i in seq_along(all_recalls)) {
  all_recalls[[i]] <- numeric(5)
}

# A list where each element is length-5 numeric (one entry per fold)
all_accuracies <- vector("list", length(k_values))
names(all_accuracies) <- paste0("k", k_values)

for (i in seq_along(all_accuracies)) {
  all_accuracies[[i]] <- numeric(5)
}

# A list where each element is length-5 numeric (one entry per fold)
all_special_measures <- vector("list", length(k_values))
names(all_special_measures) <- paste0("k", k_values)

for (i in seq_along(all_special_measures)) {
  all_special_measures[[i]] <- numeric(5)
}

fold_1 = train[train$group == 1,]
fold_2 = train[train$group == 2,]
fold_3 = train[train$group == 3,]
fold_4 = train[train$group == 4,]
fold_5 = train[train$group == 5,]

for (f in 1:5) {
  # Find our fold
  if (f == 1){
    fold = fold_1
  }
  if (f == 2){
    fold = fold_2
  }
  if (f == 3){
    fold = fold_3
  }
  if (f == 4){
    fold = fold_4
  }
  if (f == 5){
    fold = fold_5
  }
  train_f = train[!rownames(train) %in% rownames(fold), ]
  #train_f = suppressMessages(anti_join(train, fold))
  n_1 = nrow(train_f)
  n_2 = nrow(fold)
  
  # Bootstrap sampling for training based on probabilities
  train_idx <- sample(1:n_1, size = n_1, replace = TRUE, prob = train_f$w_inst)
  train_data <- train_f[train_idx,]
  
  # Select validation set (again bootstrap) from remaining observations -- We can change this to also get an accurate accuracy meassure
  #valid_idx <- sample(1:n_2, size = n_2, replace = TRUE, prob = fold$w_inst)
  #valid_data <- fold[valid_idx,]
  valid_data <- fold
  
  # Remove unwanted columns (`w_inst`, `cumulative_value`, `group`) and select features
  exclude_cols <- c("w_inst", "cumulative_value", "group")  # unwanted columns
  
  train_X <- train_data[, !(names(train_data) %in% exclude_cols)]  # Features for training
  valid_X <- valid_data[, !(names(valid_data) %in% exclude_cols)]  # Features for testing
  
  # Extract class labels (target variable)
  train_Y <- train_data$Default
  valid_Y <- valid_data$Default
  
  # Transform it so that it can be manipulated by the torch and luz packages
  
  train_X <- model.matrix(Default ~ . -1, data = train_X)
  valid_X <- model.matrix(Default ~ . -1, data = valid_X)
  
  train_Y <- unlist(as.integer(as.factor(train_Y)))
  valid_Y <- unlist(as.integer(as.factor(valid_Y)))
  
  
  # Evaluate multiple k values
  for (i in seq_along(k_values)) {
    k <- k_values[i]
    
    # Apply ANNs
    
    if (k == 1){
      modelnn = modelnn_1
    }
    if (k == 2){
      modelnn = modelnn_2
    }
    if (k == 3){
      modelnn = modelnn_3
    }
    if (k == 4){
      modelnn = modelnn_4
    }
    if (k == 5){
      modelnn = modelnn_5
    }
    if (k == 6){
      modelnn = modelnn_6
    }
    if (k == 7){
      modelnn = modelnn_7
    }
    if (k == 8){
      modelnn = modelnn_8
    }
    if (k == 9){
      modelnn = modelnn_9
    }
    if (k == 10){
      modelnn = modelnn_10
    }
    
    system.time(
      fitted <- modelnn %>%
        fit(
          data = list(train_X, train_Y), 
          epochs = 25, # 25 epochs are adequate for our learning rates
          #valid_data = list(valid_X, valid_Y), - Do not need it
          dataloader_options = list(batch_size = 256), # 256 is a good number
          verbose = TRUE
        )
    )
    
    # Confusion Matrix (ORDER INVERTED*) - Compute Accuracy, Recall and other important metrics
    
    confusion_matrix = table(as_array(torch_argmax(predict(fitted, valid_X), dim = 2)), valid_Y)
    true_pos = confusion_matrix[2, 2]
    true_neg = confusion_matrix[1, 1]
    false_neg = confusion_matrix[1, 2]
    false_pos = confusion_matrix[2, 1]
    suma = sum(confusion_matrix)
    accuracy = (true_pos + true_neg) / suma
    recall = true_pos / (true_pos + false_neg)
    special_measure = (-1*false_pos -5*false_neg) / suma
    
    # Accumulate accuracy, recall and utility metric across all folds
    all_accuracies[[i]][f] <- accuracy
    all_recalls[[i]][f] <- recall
    all_special_measures[[i]][f] <- special_measure
  }
}

# We do here a brief analysis - t-Student (df=4) for fat tails and more broad CI

cv_summary_acc <- tibble(
  k = k_values,
  mean_acc = sapply(all_accuracies, mean),
  sd_acc   = sapply(all_accuracies, sd)
) %>%
  mutate(
    # Approximate 95% CI for the mean by t-distribution (df = 4) ...
    se_acc   = sd_acc / sqrt(5),
    margin   = qt(0.975, df = 4) * se_acc,
    ci_lower = mean_acc - margin,
    ci_upper = mean_acc + margin
  )

with(cv_summary_acc, {
  plot(k, mean_acc, type = "o", pch = 16, lwd = 2,
       xlab = "k value", ylab = "Accuracy",
       ylim = c(min(ci_lower), max(ci_upper)))
  arrows(k, ci_lower, k, ci_upper, 
         code = 3, angle = 90, length = 0.05)
})

cv_summary_rec <- tibble(
  k = k_values,
  mean_rec = sapply(all_recalls, mean),
  sd_rec   = sapply(all_recalls, sd)
) %>%
  mutate(
    # Approximate 95% CI for the mean by t-distribution (df = 4) ...
    se_rec   = sd_rec / sqrt(5),
    margin   = qt(0.975, df = 4) * se_rec,
    ci_lower = mean_rec - margin,
    ci_upper = mean_rec + margin
  )

with(cv_summary_rec, {
  plot(k, mean_rec, type = "o", pch = 16, lwd = 2,
       xlab = "k value", ylab = "Recall",
       ylim = c(min(ci_lower), max(ci_upper)))
  arrows(k, ci_lower, k, ci_upper, 
         code = 3, angle = 90, length = 0.05)
})

cv_summary_spec <- tibble(
  k = k_values,
  mean_sel = sapply(all_special_measures, mean),
  sd_sel   = sapply(all_special_measures, sd)
) %>%
  mutate(
    # Approximate 95% CI for the mean by t-distribution (df = 4)
    se_sel   = sd_sel / sqrt(5),
    margin   = qt(0.975, df = 4) * se_sel,
    ci_lower = mean_sel - margin,
    ci_upper = mean_sel + margin
  )

with(cv_summary_spec, {
  plot(k, mean_sel, type = "o", pch = 16, lwd = 2,
       xlab = "k value", ylab = "Utility Measure",
       ylim = c(min(ci_lower), max(ci_upper)))
  arrows(k, ci_lower, k, ci_upper, 
         code = 3, angle = 90, length = 0.05)
})

cv_summary_acc
cv_summary_rec
cv_summary_spec # The colour red and dots are not arithmetic errors, it is just 
# the way R prints “this is a double (floating-point) value, even though it has 
# no fractional part.”

# Will select k = 6 as it has the lowest validation loss (maximal utility) and
# an adequate variance. As we can see with our other metrics, we achieve a higher
# recall than usual, 73.4%, with very low variance, and an accuracy of 67.6% with
# low variance again.