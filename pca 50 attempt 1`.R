# 1. Load the data
data <- read.csv("ohe_cleaned_data.csv")

# 2. Select only the numeric columns (i.e. your dummy variables; exclude any non‑numeric identifiers)
numeric_vars <- sapply(data, is.numeric)
pca_data      <- data[, numeric_vars]

# 3. Run PCA (centering & scaling each variable to unit variance)
pca_res <- prcomp(pca_data, center = TRUE, scale. = TRUE)

# 4. Examine the results
summary(pca_res)             # shows SD, proportion and cumulative proportion of variance
head(pca_res$rotation, 10)   # top 10 loadings per principal component

# 5. Scree plot to decide how many PCs to retain
plot(pca_res, type = "l", main = "Scree Plot")

# 6. (Optional) Biplot of PC1 vs PC2
biplot(pca_res, scale = 0, cex = 0.6)