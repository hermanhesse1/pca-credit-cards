## KNN based on 28 PCA

rm(list=ls())

train = read.csv("pca_training.csv", row.names = 1)
test = read.csv("pca_test.csv", row.names = 1)

sdevs <- sapply(train, sd)
mn <- colMeans(train)

train[,-28] = scale(train[-28], center=mn[-28], scale=sdevs[-28])
test[,-28] = scale(test[,-28], center=mn[-28], scale=sdevs[-28])

summary(train)
summary(test)


