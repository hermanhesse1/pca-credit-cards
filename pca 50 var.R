# Clear workspace
rm(list=ls())

# 0. Load packages
library(dplyr)
library(ggplot2)
library(plotly)
library(factoextra)
library(scales)    # for percent_format()

# 1. Read in the one‑hot‑encoded dataset
df <- read.csv("ohe_cleaned_data.csv")

# 2. Identify & remove target 
target_col <- names(df)[50]
y <- df[[target_col]]
features_all_enc <- df %>% select(-all_of(target_col))

# 3. Financial (continuous) subset
#    (will ignore the target automatically)
cont_df <- df
features_fin <- cont_df %>%
  select(LIMIT_BAL, AGE , starts_with("Bill.Satement"), starts_with("Amount.Paid"))

# 4. PCA on all encoded features
pca_all_enc <- prcomp(features_all_enc, center = TRUE, scale. = TRUE)
print(summary(pca_all_enc))

eig_all    <- pca_all_enc$sdev^2
var_all    <- eig_all / sum(eig_all)
cumvar_all <- cumsum(var_all)
var_table_all <- data.frame(
  PC         = seq_along(eig_all),
  Eigenvalue = eig_all,
  Proportion = var_all,
  Cumulative = cumvar_all
)
print(head(var_table_all, 10))

# 5. PCA on financial subset (continuous only)
pca_fin <- prcomp(features_fin, center = TRUE, scale. = TRUE)
print(summary(pca_fin))

eig_fin    <- pca_fin$sdev^2
var_fin    <- eig_fin / sum(eig_fin)
cumvar_fin <- cumsum(var_fin)
var_table_fin <- data.frame(
  PC         = seq_along(eig_fin),
  Eigenvalue = eig_fin,
  Proportion = var_fin,
  Cumulative = cumvar_fin
)
print(head(var_table_fin, 10))

# 6. Prepare scores + default label for plotting
data_scores <- data.frame(
  pca_all_enc$x[, 1:3],
  Default = factor(y, levels = c(0,1), labels = c("No Default","Default"))
)

# 7a. Scree Plot
scree_df <- data.frame(PC = seq_along(eig_all), Eigenvalue = eig_all)
ggplot(scree_df, aes(PC, Eigenvalue)) +
  geom_col(fill = "steelblue") +
  geom_line(aes(group=1), color = "darkblue") +
  geom_point(color = "darkblue") +
  scale_x_continuous(breaks = seq_along(eig_all)) +
  labs(title="Scree Plot (All Encoded Features)",
       x="Principal Component", y="Eigenvalue")

# 7b. Cumulative Variance Plot
cumvar_df <- data.frame(PC = seq_along(cumvar_all), CumVar = cumvar_all)
ggplot(cumvar_df, aes(PC, CumVar)) +
  geom_line(color = "darkgreen") +
  geom_point(color = "darkgreen") +
  scale_x_continuous(breaks = seq_along(cumvar_all)) +
  scale_y_continuous(labels = percent_format()) +
  geom_hline(yintercept = 0.80, linetype = "dashed", color = "gray") +
  labs(title="Cumulative Variance Explained",
       x="# PCs", y="Cumulative Variance")

# 7c. 2D PCA Scatter (PC1 vs PC2)
ggplot(data_scores, aes(PC1, PC2, color = Default)) +
  geom_point(alpha = 0.5) +
  labs(title="PCA 2D Scatter (PC1 vs PC2)",
       x="PC1", y="PC2", color="Default") +
  theme_minimal()

# 7d. Interactive 3D PCA Scatter
plot_ly(data_scores,
        x = ~PC1, y = ~PC2, z = ~PC3,
        color = ~Default, colors = c("blue","red"),
        type = "scatter3d", mode = "markers") %>%
  layout(title="3D PCA Scatter (First 3 PCs)")

# 7e. PCA Biplot
fviz_pca_biplot(pca_all_enc,
                geom.ind = "point", col.ind = data_scores$Default,
                palette = c("blue","red"), addEllipses = FALSE,
                label = "var", col.var = "black", repel = TRUE) +
  ggtitle("PCA Biplot (All Encoded Features)")
