## KNN based on 28 PCA

rm(list=ls())


library(class)
library(caret)


train = read.csv("pca_training.csv", row.names = 1)
test = read.csv("pca_test.csv", row.names = 1)

sdevs <- sapply(train, sd)
mn <- colMeans(train)

train[,-28] = scale(train[-28], center=mn[-28], scale=sdevs[-28])
test[,-28] = scale(test[,-28], center=mn[-28], scale=sdevs[-28])

summary(train)
summary(test)



train[,28]<- as.factor(train[,28])
test[,28]<- as.factor(test[,28])

## validation split

set.seed(15)
n <- nrow(train)
train_idx <- sample.int(n, size = 0.875*n) # 80% for training
val <- train[-train_idx, ]
train  <- train[train_idx, ]

## knn basic on train

# Define k‐grid
k_values <- seq(1, 501, 10)

# Preallocate vector
val_acc <- numeric(length(k_values))

# Loop to compute validation accuracy for each k
for (i in seq_along(k_values)) {
  k <- k_values[i]
  preds <- knn(
    train[ , -28],     # predictors in training
    val[   , -28],     # predictors in validation
    train[ ,  28],     # training labels
    k = k,
    use.all = F
  )
  val_acc[i] <- mean(preds == val[,28])
}

# Combine and inspect
results <- data.frame(
  k       = k_values,
  accuracy = val_acc
)
print(results)

preds <- knn3(
  
)
# Identify best k
best_idx <- which.max(results$accuracy)
best_k   <- results$k[best_idx]
best_acc <- results$accuracy[best_idx]
cat("Optimal k =", best_k, 
    "with validation accuracy =", round(best_acc, 4), "\n")



###############
k_values = seq(1, 501, 10)

cv_results <- data.frame(k = k_values, Accuracy = rep(0, length(k_values)))

basic_knn = knn(train[,-28], test[,-28] , train[,28] , k =5)

## knn basic on validation




## adjust hyper parameters, k 

## produce validation confusion matrix, 

## get confusion matrix on test


## knn with proportions on train

## use hyperparameters k (sqrt(n)) and proportion (10 different levels) 

## cross-validation for knn adjusting hyper parameters

##



?knn #have a look
is(train[,28])[1] #should be factor
predicted_knn1 = knn(train[,-28] , test[,-28] , train[,28] , k =5)
is(predicted_knn1)[1]

confusion_matrix = table(predicted_knn1, test[,28])
true_neg = confusion_matrix["No","No"]
true_pos = confusion_matrix["Yes","Yes"]
false_pos = confusion_matrix["Yes","No"]
false_neg = confusion_matrix["No","Yes"]
misclassification_rate = (false_pos + false_neg) / sum(confusion_matrix)
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg) #sensitivity
misclassification_rate
precision
recall



library(class)

train_labels <- factor(train[,28], levels = c(0, 1))
actual      <- factor(test[,28],  levels = c(0, 1))

# 2. Run kNN
predicted_knn1 <- knn(
  train[ , -28],
  test[  , -28],
  train_labels,
  k = 5, prob = T
)


vote_prop <- attr(predicted_knn1, "prob")



# 3. Build confusion matrix
confusion_matrix <- table(
  Predicted = predicted_knn1,
  Actual    = actual
)



confusion_matrix

# 4. Extract true negatives (Pred = 0, Actual = 0)
true_neg <- confusion_matrix["0", "0"]
true_neg
true_pos  <- confusion_matrix["1", "1"]
false_pos <- confusion_matrix["1", "0"]
false_neg <- confusion_matrix["0", "1"]

misclassification_rate = (false_pos + false_neg) / sum(confusion_matrix)
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg) #sensitivity
misclassification_rate
precision
recall



#####

library(class)

# 1. Run knn with prob=TRUE to get vote proportions for the winning class
knn_out  <- knn(
  train = train[ , -28],
  test  = test[  , -28],
  cl    = factor(train[,28], levels=c("0","1")),
  k     = 12,
  prob  = TRUE
)

# 2. Extract the “prob” attribute: proportion of the k votes for the predicted class
vote_prop <- attr(knn_out, "prob")

# 3. Convert that into “probability of class ‘1’”
#    – if knn_out == "1", then vote_prop is p(1); otherwise p(1)=1–vote_prop
prob_pos <- ifelse(knn_out == "1", vote_prop, 1 - vote_prop)

# 4. Apply a 0.4 threshold to declare “1” whenever p(1) ≥ 0.4
custom_pred <- factor(
  ifelse(prob_pos >= 0.2, "1", "0"),
  levels = c("0","1")
)

# 5. (Optional) confusion matrix and accuracy
cm <- table(Predicted = custom_pred, Actual = factor(test[,28], levels=c("0","1")))
accuracy <- sum(diag(cm)) / sum(cm)
cm; accuracy

precision <- cm["1","1"] / ( cm["1","1"] + cm["1","0"] )
precision

tp     <- cm["1","1"]    # true positives
fn     <- cm["0","1"]    # false negatives
recall <- tp / (tp + fn)
recall



######## classification tree

# 1. Install/load tree packages
install.packages(c("rpart", "rpart.plot"))
library(rpart)
library(rpart.plot)
library(caret)

# 2. Fit a classification tree on your training split
#    Formula: target (col 28) ~ all other PCs
tree_mod <- rpart(
  formula = train[,28] ~ .,
  data    = train,
  method  = "class",
  control = rpart.control(cp = 0.01)   # you can tune cp later
)

# 3. Visualize the tree
rpart.plot(
  tree_mod,
  type       = 2,     # type 2: label all nodes
  extra      = 104,   # show probability in leaves
  fallen.leaves = TRUE
)

# 4. Evaluate on your validation set
val_pred  <- predict(tree_mod, val, type = "class")
conf_mat  <- confusionMatrix(
  data      = val_pred,
  reference = val[,28],
  positive  = "1"
)
print(conf_mat)

# 5. (Optional) Tune complexity parameter via caret
set.seed(15)
tc <- trainControl(method = "cv", number = 5)
tuned <- train(
  x          = train[,-28],
  y          = train[,28],
  method     = "rpart",
  trControl  = tc,
  tuneLength = 10
)
print(tuned)
# Plot cp vs. accuracy
plot(tuned)

# 6. Re‐plot best tree
best_tree <- tuned$finalModel
rpart.plot(best_tree, type = 2, extra = 104, fallen.leaves = TRUE)

# 7. Finally, apply to your true test set
test_pred <- predict(best_tree, test, type = "class")
test_cm   <- confusionMatrix(test_pred, test[,28], positive = "1")
print(test_cm)

