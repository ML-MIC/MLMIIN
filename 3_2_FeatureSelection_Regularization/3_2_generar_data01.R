rm(list = ls())
library(caret)
library(tidyverse)
library(caret)
library(GGally)
library(kernlab)

# Set working directory
setwdOK <- try(setwd(dirname(rstudioapi::getActiveDocumentContext()$path)))

# Training and test sizes
N_tr <- 1000
N_ts <-  250

N <- N_tr + N_ts

# Replicability
set.seed(20230214)

# Number of predictor variables
numVar <- 11

# Create an empty tibble to store the data
fdata <- list_cbind(map(.x = paste0("X", 1:numVar), 
               .f = ~ tibble(X = rnorm(N), .name_repair = function(u){.x})))

# Data generation

# X1, X5, X6, X9 and X10 are the important independent variables

fdata <- fdata %>% 
  mutate(
    # X2, X3 and X7 are correlated variables
    X2 = X1 * 2 + 1.5 * rnorm(N),
    X3 = 6 * X9 + 1.5 * rnorm(N),
    X7 = X6 + 1.25 * rnorm(N),
    # Output, numeric version 
    Ynum = sin(X1) - X5^3 / 2 + exp(X6) + X9^2/4 - X10,
    # Ynum = sin(X1) - X5^3 / 2 + exp(X6) + log(abs(X9)) - X10,
    # Output, factor
    # Y = between(Ynum, quantile(Ynum, probs=0.25), quantile(Ynum, probs=0.75)),
    # Y = factor(Y, labels = c("NO", "YES")),
    
    # For regression examples
    Y = Ynum,
    Ynum = NULL
    
  )


# First look at the data
skimr::skim(fdata)


ggplot(fdata) + 
  geom_point(aes(X1, Y))

boxplot(fdata$Y)

fdata$Y[which.max(fdata$Y)] = mean(fdata$Y)

ggplot(fdata) + 
  geom_point(aes(X1, Y))



# Save the complete data set
write_csv(x = fdata, file = "3_2_data01.csv")
