##### Deep Learning - Feedforward Neural Networks #####

### Final Approach: Try to get a better recall by making our own loss function
#rm(list=ls())

## Libraries and Seed

library(torch)
library(luz) # High-level interface for torch
library(pROC)
torch_manual_seed(10)

## Preliminary Data Transformations

data = read.csv('cleaned_data.csv')

n <- nrow(data)

summary(data)

y = data$Default # Save it for later

variable_names <- c("SEX", "EDUCATION", "MARRIAGE", "Default")

for (i in variable_names) {
  data[[i]] <- as.factor(data[[i]])
}

summary(data)

#levels(data$SEX)
#levels(data$EDUCATION)
#levels(data$MARRIAGE)
#levels(data$Default)

# The following command automatically applies 1-hot encoding, and also splits
# out the features and the target. It also returns a matrix rather than a dataframe
# for the input variables. This is needed for torch, which does not work directly
# with R dataframes.

x <- model.matrix(Default ~ . -1, data = data)
p = ncol(x)
summary(x)

# We take out one of the sex variables since it does not give us any extra information (even though it does not matter in this case)

x = x[,-3]
p = ncol(x)
summary(x)

# Training/test split (Same as before cause of the seed (10))

set.seed(10)
ntest <- trunc(n / 5)
testid <- sample(1:n, ntest)
test_x = x[testid,]
test_y = y[testid]
train_x = x[-testid,]
train_y = y[-testid]

# Standardise the training data and then the test data with the same transforation

sdevs = sapply(as.data.frame(train_x), sd)
mn = colMeans(train_x)

train_x = scale(train_x, center = mn, scale = sdevs)
test_x = scale(test_x, center = mn, scale = sdevs)

# Do a little transformation for R syntax

test_y <- unlist(as.integer(as.factor(test_y)))
train_y <- unlist(as.integer(as.factor(train_y)))

# Validation split (Leakage for the validation/too small and unimportant)

n_1 = nrow(train_x)
nval <- trunc(n_1 / 4)
valid <- sample(1:nval)
valid_x = train_x[valid,]
valid_y = train_y[valid]
ttrain_x = train_x[-valid,]
ttrain_y = train_y[-valid]

## Building the Network

# Initialize the Network

modnn <- nn_module(
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

## We test the different loss functions on the validation set and choose the best one

#--> Make our own loss

# It is built in "nn_cross_entropy_loss" already!

w_values <- seq(1, 10, by = 0.5)

accuracies <- seq(1, 10, by = 0.5)

recalls <- seq(1, 10, by = 0.5)

special_measures <- seq(1, 10, by = 0.5)

for (i in 1:length(w_values)) {
  class_weights = c(1, w_values[i])
  
  w_tensor <- torch_tensor(class_weights, dtype = torch_float())
  
  #<--
  
  modelnn <- modnn %>%
    setup(
      loss = nn_cross_entropy_loss(weight = w_tensor),
      optimizer = function(params) optim_rmsprop(params, lr = 0.001),
      metrics = list(luz_metric_accuracy())
    )
  
  # Train the model
  
  system.time(
    fitted <- modelnn %>%
      fit(
        data = list(ttrain_x, ttrain_y), 
        epochs = 25, # 25 epochs are adequate for our learning rates
        #valid_data = list(valid_x, valid_y), - Do not need it
        dataloader_options = list(batch_size = 256), # 256 is a good number
        verbose = TRUE
      )
  )
  
  # Time to validate - Get the confusion matrix (ORDER INVERTED*)
  
  print(w_values[i])
  
  confusion_matrix = table(as_array(torch_argmax(predict(fitted, valid_x), dim = 2)), valid_y); print(confusion_matrix)
  true_pos = confusion_matrix[2, 2]
  true_neg = confusion_matrix[1, 1]
  false_neg = confusion_matrix[1, 2]
  false_pos = confusion_matrix[2, 1]
  sum_conf = sum(confusion_matrix)
  accuracy = (true_pos + true_neg) / sum_conf
  misclassification_rate = (false_pos + false_neg) / sum_conf
  precision = true_pos / (true_pos + false_pos)
  other_precision = true_neg / (true_neg + false_neg) # W.r.t the other class
  recall = true_pos / (true_pos + false_neg)
  selectivity = true_neg / (true_neg + false_pos)
  special_measure = (-1*false_pos -5*false_neg) / sum_conf
  accuracies[i] = accuracy
  print(accuracy)
  print(misclassification_rate)
  print(precision)
  print(other_precision)
  recalls[i] = recall
  print(recall)
  print(selectivity)
  special_measures[i] = special_measure
  print(special_measure)
}

## Let's plot it

plot(w_values, recalls, type = "b", pch = 16, col = "steelblue",
     ylim = range(c(0, 1)),
     xlab = "Weight Values", ylab = "Recall and Accuracy",
     main = "Recall and Accuracy")
lines(w_values, accuracies, type = "b", pch = 17, col = "firebrick", lty = 2)
legend("topright",
       legend = c("Recall", "Accuracy"),
       col    = c("steelblue", "firebrick"),
       pch    = c(16, 17),
       lty    = c(1, 2))

plot(w_values, special_measures, type = "b", pch = 16, col = "purple",
     xlab = "Weight Values", ylab = "Utility Measure",
     main = "Utility Measure")

special_measures

## -- Time to test - Optimal was weight 6.5 (No surprise)

class_weights = c(1, 6.5)

w_tensor <- torch_tensor(class_weights, dtype = torch_float())

modelnn <- modnn %>%
  setup(
    loss = nn_cross_entropy_loss(weight = w_tensor),
    optimizer = function(params) optim_rmsprop(params, lr = 0.001),
    metrics = list(luz_metric_accuracy())
  )

# Train the model in all the training set

system.time(
  fitted <- modelnn %>%
    fit(
      data = list(train_x, train_y), 
      epochs = 25, # As always
      dataloader_options = list(batch_size = 256), # As always
      verbose = TRUE
    )
)

# Gets the true classes from all observations in test_ds.
#truth <- sapply(seq_along(dig_test), function(x) test_ds[x][[2]])

## Get the confusion matrix (ORDER INVERTED*)

confusion_matrix = table(as_array(torch_argmax(predict(fitted, test_x), dim = 2)), test_y); confusion_matrix
true_pos = confusion_matrix[2, 2]
true_neg = confusion_matrix[1, 1]
false_neg = confusion_matrix[1, 2]
false_pos = confusion_matrix[2, 1]
sum_conf = sum(confusion_matrix)
accuracy = (true_pos + true_neg) / sum_conf
misclassification_rate = (false_pos + false_neg) / sum_conf
precision = true_pos / (true_pos + false_pos)
other_precision = true_neg / (true_neg + false_neg) # W.r.t the other class
recall = true_pos / (true_pos + false_neg)
selectivity = true_neg / (true_neg + false_pos)
accuracy
misclassification_rate
precision
other_precision
recall
selectivity

# Special Measure

special_measure = (-1*false_pos -5*false_neg) / sum_conf
special_measure

# F-1 Score

f1_score   <- 2 * precision * recall / (precision + recall)
f1_score

## Kappa Statistic

# Replace these with your numbers
TP <- true_pos
TN <- true_neg
FP <- false_pos
FN <- false_neg

# Total
N <- TP + TN + FP + FN

# Observed accuracy
Po <- (TP + TN) / N

# Marginals
p_pred_pos <- (TP + FP) / N
p_pred_neg <- (TN + FN) / N
p_act_pos  <- (TP + FN) / N
p_act_neg  <- (TN + FP) / N

# Expected agreement
Pe <- p_pred_pos * p_act_pos + p_pred_neg * p_act_neg

# Cohen's kappa
kappa <- (Po - Pe) / (1 - Pe)

# Print
cat("Cohen's kappa =", round(kappa, 3), "\n")


## ROC Curve

# 1. Get the raw logits (a torch_tensor) from your model
logits <- predict(fitted, test_x)

# 2. Convert logits → probabilities for the “Default = 1” class
#    nnf_softmax applies softmax along the class dimension (dim = 2)
prob_tensor <- nnf_softmax(logits, dim = 2)
probs       <- as_array(prob_tensor)[, 2]

# 3. Plot ROC curve + compute AUC
roc_obj <- roc(response = test_y, predictor = probs)
plot(roc_obj, main = sprintf("ROC Curve (AUC = %.3f)", auc(roc_obj)))
cat("Test AUC:", round(auc(roc_obj), 3), "\n\n")