rm(list = ls())

data = read.csv("cleaned_data.csv")

data[,c(2,3,4,6,7,8,9,10,11,24)] = lapply(data[c(2,3,4,6,7,8,9,10,11,24)], as.factor)


summary(data)


library(caret)

fac_cols <- c(2,3,4)
fac_names <- names(data)[fac_cols]

# set up the dummy‐var transformer
dv <- dummyVars(~ ., data = data[fac_cols], fullRank = T)

# create the dummy data frame
dummies <- predict(dv, newdata = data)
dummies <- as.data.frame(dummies)

# combine with the non‐factor columns
data_ohe <- cbind(data[ , -fac_cols], dummies)

write.csv(data_ohe, "ohe28_cleaned_data.csv", row.names = FALSE)




