

# Full tuning loop: utility, accuracy, recall for k = 1, 51, …, 501
rm(list=ls())

setwd("~/Masters/MA429/project/summative/default+of+credit+card+clients")

library(kknn)
library(caret)

# 1. Load & scale
train <- read.csv("pca_training.csv", row.names=1)
test  <- read.csv("pca_test.csv",     row.names=1)

sdevs <- sapply(train, sd); means <- colMeans(train)
train[,1:27] <- scale(train[,1:27], center=means[1:27], scale=sdevs[1:27])
test [,1:27] <- scale(test [,1:27], center=means[1:27], scale=sdevs[1:27])

names(train)[28] <- names(test)[28] <- "Class"
train$Class <- factor(train$Class); test$Class <- factor(test$Class)

# 2. Split 12.5% for validation
set.seed(15)
idx   <- sample(nrow(train), size = 0.875 * nrow(train))
val   <- train[-idx, ]
train <- train[ idx, ]

# 3. Define grid and storage
k_values   <- seq(1, 51, 2) # test for k between 1 to 501 using increments of 50; found that the highest utility score was between 1 to 51, so then ran again to find within this range what the utility score is
n_k        <- length(k_values)
total_util <- numeric(n_k)
accuracy   <- numeric(n_k)
recall     <- numeric(n_k)

# 4. Utility function
utility <- function(true, pred) {
  if      (true=="1" && pred=="1")  1
  else if (true=="0" && pred=="1") -1
  else if (true=="0" && pred=="0")  1
  else if (true=="1" && pred=="0") -10
  else NA
}

# 5. Loop over k
for (i in seq_along(k_values)) {
  k <- k_values[i]
  
  # fit k-NN on predictors 1:7
  fit <- kknn(
    Class ~ .,
    train = train[, c(1:15,28)],
    test  = val  [, c(1:15,28)],
    k     = k,
    kernel= "rectangular"
  )
  
  preds <- fitted(fit)
  
  # utility
  total_util[i] <- sum(mapply(utility,
                              as.character(val$Class),
                              as.character(preds)))
  
  # confusion matrix
  cm <- confusionMatrix(preds, val$Class, positive="1")
  accuracy[i] <- cm$overall["Accuracy"]
  recall[i]   <- cm$byClass ["Sensitivity"]
}

# 6. Results
results <- data.frame(
  k           = k_values,
  total_util  = total_util,
  Accuracy    = as.numeric(accuracy),
  Recall      = as.numeric(recall)
)
print(results)

test_result = predict(kknn(Class ~ .,
                    train = train[, c(1:15,28)],
                    test  = test[, c(1:15,28)],
                    k     = 9,
                    kernel= "rectangular"))

test_cm = confusionMatrix(test_result, test$Class, positive = "1")

test_cm$overall["Accuracy"]
test_cm$byClass ["Sensitivity"]
test_util <- sum(mapply(utility,
                           as.character(test$Class),
                           as.character(test_result)))

test_util
############ KNN with proportions 
