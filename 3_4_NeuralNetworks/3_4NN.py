

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import string

import pickle

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

#import joblib
import warnings

import networkx as nx



X, Y = make_circles(n_samples=1000, factor=.2, noise=0.15, random_state=2024)

inputs = ["X" + str(k) for k in range(X.shape[1])]
output = "Y"

XTR, XTS, YTR, YTS = train_test_split(X, Y,
                                      test_size=0.2,  # percentage preserved as test data
                                      random_state=1, # seed for replication
                                      stratify = Y)   # Preserves distribution of y

dfTR = pd.DataFrame(XTR, columns=inputs)
dfTR[output] = YTR
dfTS = pd.DataFrame(XTS, columns=inputs)
dfTS[output] = YTS



# %%
# Create and train the MLPClassifier
mlp_binary = MLPClassifier(hidden_layer_sizes=(8, 6), activation="relu", max_iter=10000, random_state=2024)
mlp_binary.fit(XTR, YTR)

model = mlp_binary
model_name = "mlp_binary"

# %%
make_circles_model = {'mlp_binary':mlp_binary}

# %%
[layer.shape for layer in model.coefs_]

# %%
model.coefs_[0]




# %%
# Dataset for Training Predictions
dfTR_eval = dfTR.copy()
# Store the actual predictions
newCol = 'Y_'+ model_name +'_prob_neg'; 
dfTR_eval[newCol] = model.predict_proba(XTR)[:, 0]
newCol = 'Y_'+ model_name +'_prob_pos'; 
dfTR_eval[newCol] = model.predict_proba(XTR)[:, 1]
newCol = 'Y_'+ model_name +'_pred'; 
dfTR_eval[newCol] = model.predict(XTR)

# %%
# Test predictions dataset
dfTS_eval = dfTS.copy()
newCol = 'Y_'+ model_name +'_prob_neg'; 
dfTS_eval[newCol] = model.predict_proba(XTS)[:, 0]
newCol = 'Y_'+ model_name +'_prob_pos'; 
dfTS_eval[newCol] = model.predict_proba(XTS)[:, 1]
newCol = 'Y_'+ model_name +'_pred'; 
dfTS_eval[newCol] = model.predict(XTS)

# This generates the confusion matrices and shows that classification is almost perfect both in training and test.

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
fig = plt.figure(constrained_layout=True, figsize=(6, 2))
spec = fig.add_gridspec(1, 3)
ax1 = fig.add_subplot(spec[0, 0]);ax1.set_title('Training'); ax1.grid(False)
ax2 = fig.add_subplot(spec[0, 2]);ax2.set_title('Test'); ax2.grid(False)
ConfusionMatrixDisplay.from_estimator(model, XTR, YTR, cmap="Greens", colorbar=False, ax=ax1, labels=[1, 0])
ConfusionMatrixDisplay.from_estimator(model, XTS, YTS, cmap="Greens", colorbar=False, ax=ax2, labels=[1, 0])
plt.suptitle("Confusion Matrices for "+ model_name)
plt.show(); 

# %% [markdown]
# The ROC curves tell a similar story. We satisfy ourselves that this looks good enough, and will not spend more time on performance measures. Feel free to dig deeper, you have all the tools from previous sessions. 

# %%
from sklearn.metrics import RocCurveDisplay
fig = plt.figure(figsize=(12, 4))
spec = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(spec[0, 0]);ax1.set_title('Training')
ax2 = fig.add_subplot(spec[0, 1]);ax2.set_title('Test')
RocCurveDisplay.from_estimator(model, XTR, YTR, plot_chance_level=True, ax=ax1)
RocCurveDisplay.from_estimator(model, XTS, YTS, plot_chance_level=True, ax=ax2);
plt.suptitle("ROC Curves for "+ model_name)
plt.show(); 

# %% [markdown]
# ---

# %% [markdown]
# ## Using Torch (pytorch) to fit the neural network

# %%
XTR.mean(axis=0), XTR.std(axis=0)

# %%
import torch

XTR_tensors =  torch.from_numpy(XTR)
YTR_tensors =  torch.from_numpy(YTR)

XTS_tensors =  torch.from_numpy(XTS)
YTS_tensors =  torch.from_numpy(YTS)

# %%
# Set XTR to be float32
XTR_tensors = XTR_tensors.to(torch.float32)
YTR_tensors = YTR_tensors.to(torch.float32)
XTS_tensors = XTS_tensors.to(torch.float32)
YTS_tensors = YTS_tensors.to(torch.float32)


# %%
import torch
import torch.nn as nn

# Explicitly setting the seed for reproducibility (random_state=2024)
torch.manual_seed(2024)

# Defining the architecture
mlp_binary = nn.Sequential(
    # Layer 1: 2 inputs -> 8 neurons
    nn.Linear(2, 8),
    nn.ReLU(),
    
    # Layer 2: 8 neurons -> 6 neurons
    nn.Linear(8, 6),
    nn.ReLU(),
    
    # Layer 3: 6 neurons -> 1 output
    nn.Linear(6, 1),
    nn.Sigmoid()
)

print(mlp_binary)

# %%
# 1. Loss function (Binary Cross Entropy)
loss_fn = nn.BCELoss() 

# 2. Optimizer (Adam is the scikit-learn default)
optimizer = torch.optim.Adam(mlp_binary.parameters(), lr=0.001)

# 3. Example Forward Pass
# X_train is a tensor of shape (batch_size, 2)
# outputs = mlp_binary(X_train)

# %%
from torch.utils.data import TensorDataset

# Convert to Tensors
train_ds = TensorDataset(XTR_tensors, YTR_tensors)

test_ds = TensorDataset(XTS_tensors, YTS_tensors)

# %%
from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

# %% [markdown]
# Let us get one tensor out of train_ds

# %%
X0, Y0 = train_ds[0]
X0, Y0

# %%
Y0hat = model(X0)
Y0hat, Y0hat.shape, Y0, Y0.shape


# %%
Y0 = Y0.unsqueeze(-1)
print(Y0hat.shape, Y0.shape)

# %%
loss_fn(model(X0), Y0)

# %%
import datetime

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):

    # Loop over epochs
    for epoch in range(1, n_epochs + 1):  
        
        start_time = datetime.datetime.now()
        loss_train = 0.0
        
        # Loop over batches
        for x, y in train_loader:  
            
            # forward pass
            outputs = model(x)
          
            # compute loss
            loss = loss_fn(outputs, y.unsqueeze(-1))

            # reset gradients before backpropagation
            optimizer.zero_grad()
            
            # backpropagation
            loss.backward()
            
            # update parameters
            optimizer.step()

            # update training loss
            loss_train += loss.item()

        end_time = datetime.datetime.now()
        epoch_duration = (end_time - start_time).total_seconds()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {:.6f}, Time {:.2f}s'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader), epoch_duration))

# %%
training_loop(
    n_epochs = 100,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
)


