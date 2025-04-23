# Clear workspace
rm(list=ls())

# 0. Load packages
library(dplyr)
library(ggplot2)
library(plotly)
library(factoextra)
library(scales)    # for percent_format()

# 1. Read in the one‑hot‑encoded dataset
df <- read.csv("ohe28_cleaned_data.csv")

# 2. Identify & remove target 
target_col <- names(df)[21]
y <- df[[target_col]]
features_all_enc <- df %>% select(-all_of(target_col))

set.seed(10)
n <- nrow(df)
train_idx <- sample.int(n, size = 0.8*n) # 80% for training
train <- df[train_idx, ]
test  <- df[-train_idx, ]

write.csv(train, "train.csv")
write.csv(test, "test.csv")


## PCA based on train set only

pca_train <- prcomp(train[,-21], center = T, scale. = T)
pca_test <- predict(pca_train, newdata=test[,-21])


summary(pca_train)

pca_train_df <- as.data.frame(pca_train$x)

pca_test_df <- as.data.frame(pca_test)

pca_train_df[,28] <-train[,21]
pca_test_df[,28] <-test[,21]

write.csv(pca_train_df, "pca_training.csv")
write.csv(pca_test_df, "pca_test.csv")


