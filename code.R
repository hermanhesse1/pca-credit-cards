# https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

setwd("~/Masters/MA429/project/summative/default+of+credit+card+clients")

rm(list=ls())

library(ggplot2)
library(tidyverse)

data = read.csv("dataset.csv")

# column_names = data[1,]

# data = data[-1,]


colnames(data)



data[] = lapply(data, as.numeric)

data = data[,-1]

colnames(data)[6] = "PAY_1"

data[, 6:11] <- lapply(data[, 6:11], function(x) 
  ifelse(x >= 4, 3, x)
)

data[, 3] <- ifelse(data[, 3] >= 4, 0, data[, 3])


data[,c(2,3,4,6,7,8,9,10,11,24)] = lapply(data[c(2,3,4,6,7,8,9,10,11,24)], as.factor)

summary(data)

data = data[ , c(1:5, 11:6, 17:12, 23:18, 24)]

colnames(data)[c(6:11)] = c("Repayment status April 2005", "Repayment status May 2005", "Repayment status June 2005", "Repayment status July 2005", "Repayment status August 2005", "Repayment status September 2005")
                            
colnames(data)[c(12:17)] = c("Bill Satement April 2005", "Bill Satement May 2005", "Bill Satement June 2005", "Bill Satement July 2005", "Bill Satement August 2005", "Bill Satement September 2005")

colnames(data)[c(18:23)] = c("Amount Paid April 2005", "Amount Paid May 2005", "Amount Paid June 2005", "Amount Paid July 2005", "Amount Paid August 2005", "Amount Paid September 2005")

colnames(data)[24] = "Default"

summary(data)




plot(data$`Repayment status April 2005`)


levels(data$`Repayment status April 2005`)
levels(data$`Repayment status May 2005`)
levels(data$`Repayment status June 2005`)
levels(data$`Repayment status July 2005`)
levels(data$`Repayment status August 2005`)
levels(data$`Repayment status September 2005`)



plot(data$`Repayment status September 2005`)


options(scipen = 999)

write.csv(data, "cleaned_data.csv", row.names = FALSE)





                  