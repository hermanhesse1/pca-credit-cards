##### Deep Learning - Feedforward Neural Networks #####

### Better Approach: Try to get a better recall by bootstrapping

#rm(list=ls())

## Libraries and Seed

library(torch)
library(luz) # high-level interface for torch
torch_manual_seed(46)

## Preliminary Data Transformations

data = read.csv('cleaned_data.csv')

n <- nrow(data)

summary(data)

y = data$Default # Save it for later

# --> Brief analysis of the Kappa Static



# <--

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

# Training/test split

set.seed(46)
ntest <- trunc(n / 5)
testid <- sample(1:n, ntest)
test_x = x[testid,]
test_y = y[testid]
train_x = x[-testid,]
train_y = y[-testid]

#--> Preparing Bootstrap - OPTIONAL

#train_x = as.data.frame(train_x)

#n_t = nrow(train_x)

#train_x$w_inst = train_y

#for (i in 1:n_t){
#  if (train_x$w_inst[i] == 1){
#    train_x$w_inst[i] = 5 # Will give approximately a 50/50 split between the classes
#  }
#  else {train_x$w_inst[i] = 1}
#}

#train_idx = sample(1:nrow(train_x), size = nrow(train_x), replace = TRUE, prob = train_x$w_inst)

#train_x <- train_x[train_idx,]
#train_y <- train_y[train_idx]

#train_x <- model.matrix(w_inst ~ . -1, data = train_x)

#<--

# Standardise the training data and then the test data with the same transforation

sdevs = sapply(as.data.frame(train_x), sd)
mn = colMeans(train_x)

train_x = scale(train_x, center = mn, scale = sdevs)
test_x = scale(test_x, center = mn, scale = sdevs)

test_y <- unlist(as.integer(as.factor(test_y)))
train_y <- unlist(as.integer(as.factor(train_y)))

# Do a little tranformation for R syntax

test_y <- unlist(as.integer(as.factor(test_y)))
train_y <- unlist(as.integer(as.factor(train_y)))

## Building the Network

# Initialize the Network

modnn_3 <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(in_features = 27, out_features = 50)
    self$linear2 <- nn_linear(in_features = 50, out_features = 100)
    self$linear3 <- nn_linear(in_features = 100, out_features = 50)
    self$linear4 <- nn_linear(in_features = 50, out_features = 2)
    self$activation <- nn_relu()
    self$dropout1 <- nn_dropout(0.2) # 0.2
    self$dropout2 <- nn_dropout(0.35) # 0.35
    self$dropout3 <- nn_dropout(0.5) # 0.5
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

#--> Make our own loss

# It is built in "nn_cross_entropy_loss" already!

class_weights = c(1, 6) # (1, 6)

w_tensor <- torch_tensor(class_weights, dtype = torch_float())

#<--

modelnn_3 <- modnn_3 %>%
  setup(
    loss = nn_cross_entropy_loss(weight = w_tensor),
    optimizer = function(params) optim_rmsprop(params, lr = 0.001), # 0.001
    metrics = list(luz_metric_accuracy())
  )

# Train the model

system.time(
  fitted <- modelnn_3 %>%
    fit(
      data = list(train_x, train_y), 
      epochs = 25, # 25
      valid_data = 0.2,
      dataloader_options = list(batch_size = 1000), # 1000
      verbose = TRUE
    )
)

# Show results in a plot

plot(fitted)

## Time to test

# Define accuracy

accuracy <- function(pred, truth) {
  mean(pred == truth) }

# Gets the true classes from all observations in test_ds.
#truth <- sapply(seq_along(dig_test), function(x) test_ds[x][[2]])

# Get the accuracy

fitted %>% 
  predict(test_x) %>% 
  torch_argmax(dim = 2) %>%  # The predicted class is the one with higher 'logit'.
  as_array() %>% # We convert to an R object
  accuracy(test_y)

# Get the confusion matrix

confusion_matrix = table(as_array(torch_argmax(predict(fitted, test_x), dim = 2)), test_y); confusion_matrix
true_neg = confusion_matrix[2, 2]
true_pos = confusion_matrix[1, 1]
false_pos = confusion_matrix[1, 2]
false_neg = confusion_matrix[2, 1]
misclassification_rate = (false_pos + false_neg) / sum(confusion_matrix)
precision = true_pos / (true_pos + false_pos)
other_precision = true_neg / (true_neg + false_neg) # W.r.t the other class
recall = true_pos / (true_pos + false_neg)
selectivity = true_neg / (true_neg + false_pos)
misclassification_rate
precision
other_precision
recall
selectivity