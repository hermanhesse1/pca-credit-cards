# Advanced Data Quality Check & PCA Analysis with One‑Hot Encoding

# 0. Load required packages
# Uncomment install.packages() if any package is missing
# install.packages(c("readxl", "dplyr", "ggplot2", "plotly", "factoextra", "scales"))
library(readxl)     # read_excel()
library(dplyr)      # data manipulation\llibrary(ggplot2)    # plotting\llibrary(plotly)     # interactive 3D plots\llibrary(factoextra) # PCA biplot\llibrary(scales)     # percentage scales

# 1. Read & Prepare the Data
df <- read_excel("default of credit card clients.xls", skip = 1) %>%
  rename(Y = `default payment next month`)

# Drop ID if present
if("ID" %in% names(df)) df <- select(df, -ID)

# 2. Basic Structure Checks
str(df)
dim(df)    # should be 30000 x 24
names(df)  # view column names

# 3. Check for missing values
cat("Total NAs:", sum(is.na(df)), "\n")
cat("Any NAs present?", anyNA(df), "\n")
colSums(is.na(df))  # expect all zeros

# 4. Verify repayment‐history codes (–2 to 8)
repay_cols <- c("PAY_0", paste0("PAY_", 2:6))
sapply(df[ , repay_cols], function(x) c(min = min(x), max = max(x)))

# 5. Check ranges of continuous variables
cat("LIMIT_BAL:")     ; print(summary(df$LIMIT_BAL))
cat("AGE:")           ; print(summary(df$AGE))
cat("BILL_AMT1–6:")   ; print(summary(select(df, starts_with("BILL_AMT"))))
cat("PAY_AMT1–6:")    ; print(summary(select(df, starts_with("PAY_AMT"))))

# 6. Recode & Factor Categorical Variables
# Based on documented categories, grouping "others" as needed
f_cat <- df %>%
  mutate(
    SEX = factor(SEX, levels = c(1,2), labels = c("Male","Female")),
    EDUCATION = case_when(
      EDUCATION == 1 ~ "Graduate school",
      EDUCATION == 2 ~ "University",
      EDUCATION == 3 ~ "High school",
      TRUE ~ "Others"
    ) %>% factor(levels = c("Graduate school","University","High school","Others")),
    MARRIAGE = case_when(
      MARRIAGE == 1 ~ "Married",
      MARRIAGE == 2 ~ "Single",
      MARRIAGE == 3 ~ "Divorce",
      TRUE ~ "Others"
    ) %>% factor(levels = c("Married","Single","Divorce","Others"))
  )
# Overwrite df with factored categories
df <- f_cat

# 7. One‐Hot Encode Categorical Variables
# model.matrix() creates dummy columns for each factor level
cat_matrix <- model.matrix(~ SEX + EDUCATION + MARRIAGE - 1, data = df)
# Confirm dummy columns
colnames(cat_matrix)

# 8. Prepare Continuous Features (exclude categorical & target)
cont_df <- df %>% select(-Y, -SEX, -EDUCATION, -MARRIAGE)

# 9. Combine Encoded & Continuous Features for PCA
features_all_enc <- cbind(cont_df, cat_matrix)

# 10. PCA on All Encoded Features
pca_all_enc <- prcomp(features_all_enc, center = TRUE, scale. = TRUE)
summary(pca_all_enc)

# Extract eigenvalues & explained variance
eig_all <- pca_all_enc$sdev^2
var_all <- eig_all / sum(eig_all)
cumvar_all <- cumsum(var_all)
var_table_all <- data.frame(
  PC = 1:length(eig_all),
  Eigenvalue = eig_all,
  Proportion = var_all,
  Cumulative = cumvar_all
)
print(head(var_table_all, 10))  # first 10 PCs

# 11. PCA on Financial Subset (CONTINUOUS ONLY)
features_fin <- cont_df %>% select(LIMIT_BAL, AGE, starts_with("BILL_AMT"), starts_with("PAY_AMT"))
pca_fin <- prcomp(features_fin, center = TRUE, scale. = TRUE)
summary(pca_fin)

eig_fin <- pca_fin$sdev^2
var_fin <- eig_fin / sum(eig_fin)
cumvar_fin <- cumsum(var_fin)
var_table_fin <- data.frame(
  PC = 1:length(eig_fin),
  Eigenvalue = eig_fin,
  Proportion = var_fin,
  Cumulative = cumvar_fin
)
print(head(var_table_fin, 10))

# 12. Visualizations (using pca_all_enc)
# Prepare scores + default label
data_scores <- data.frame(pca_all_enc$x[,1:3],
                          Default = factor(df$Y, labels = c("No Default","Default")))

# 12a. Scree Plot
scree_df <- data.frame(PC = 1:length(eig_all), Eigenvalue = eig_all)
ggplot(scree_df, aes(x = PC, y = Eigenvalue)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_line(group=1, color="darkblue") + geom_point(color="darkblue") +
  scale_x_continuous(breaks=1:length(eig_all)) +
  labs(title="Scree Plot (All Encoded Features)", x="Principal Component", y="Eigenvalue")

# 12b. Cumulative Variance Plot
ggplot(data.frame(PC = 1:length(eig_all), CumVar = cumvar_all), aes(x=PC, y=CumVar)) +
  geom_line(color="darkgreen") + geom_point(color="darkgreen") +
  scale_x_continuous(breaks=1:length(eig_all)) +
  scale_y_continuous(labels = percent_format()) +
  geom_hline(yintercept=0.80, linetype="dashed", color="gray") +
  labs(title="Cumulative Variance Explained", x="# PCs", y="Cumulative Variance")

# 12c. 2D PCA Scatter (PC1 vs PC2)
ggplot(data_scores, aes(x=PC1, y=PC2, color=Default)) +
  geom_point(alpha=0.5) +
  labs(title="PCA 2D Scatter (PC1 vs PC2)", x="PC1", y="PC2", color="Default") +
  theme_minimal()

# 12d. Interactive 3D PCA Scatter (PC1, PC2, PC3)
plot_ly(data_scores, x=~PC1, y=~PC2, z=~PC3, color=~Default,
        colors = c("blue","red"), type="scatter3d", mode="markers") %>%
  layout(title="3D PCA Scatter (First 3 PCs)")

# 12e. PCA Biplot
fviz_pca_biplot(pca_all_enc,
                geom.ind = "point", col.ind = data_scores$Default,
                palette = c("blue","red"), addEllipses = FALSE,
                label = "var", col.var = "black", repel = TRUE) +
  ggtitle("PCA Biplot (All Encoded Features)")

# End of Script


