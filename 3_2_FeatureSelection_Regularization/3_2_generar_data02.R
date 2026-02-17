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
numVar <- 10

# Create an empty tibble to store the data
fdata <- list_cbind(map(.x = paste0("X", 1:numVar), 
               .f = ~ tibble(X = rnorm(N), .name_repair = function(u){.x})))

# Data generation

# X1, X5, X6, X9 and X10 are the important independent variables

fdata <- fdata %>% 
  mutate(
    X1 = rnorm(N),
    X2 = X1 + rnorm(N),
    X3 = - X1 + rnorm(N),
    X4 = rnorm(N),
    X5 = X6 + rnorm(N),
    X6 = rnorm(N),
    X7 = X8 + rnorm(N),
    X8 = rnorm(N),
    X9 = X6 +  rnorm(N),
    X10 = rnorm(N),
    # Output, numeric version 
    Ynum = 4 + X1 + X4 + X8 + 10000 * X10 + 100 * rnorm(N),
    # Output, factor
    # Y = between(Ynum, quantile(Ynum, probs=0.25), quantile(Ynum, probs=0.75)),
    # Y = factor(Y, labels = c("NO", "YES")),
    
    # For regression examples
    Y = Ynum,
    Ynum = NULL
    
  )


# First look at the data
skimr::skim(fdata)

# ggpairs(fdata)

# Save the complete data set
write_csv(x = fdata, file = "3_2_data02.csv")
