# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from msvr import kernelmatrix
from msvr import msvr
import numpy as np


# %%
x1 = np.sin(np.arange(0, 9, 0.01))
x2 = np.cos(np.arange(0, 9, 0.01))
x3 = x1**2
x4 = (x1+x2)/2

x = np.vstack((x1,x2)).T
y = np.vstack((x3,x4)).T


# %%
# Input & Output
# Xtrain: number of samples * input dimension
# Ytrain: number of samples * output dimension

Xtrain = x[:600, :]
Ytrain = y[:600, :]
Xtest = x[600:, :]
Ytest = y[600:, :]
Xtrain = (Xtrain-np.min(Xtrain))/(np.max(Xtrain)-np.min(Xtrain))
Ytrain = (Ytrain-np.min(Ytrain))/(np.max(Ytrain)-np.min(Ytrain))


# %%
# Parameters
ker     = 'rbf'
C       = 2
epsi = 0.001
par   = 0.5
tol     = 1e-10


# %%
# Train
Beta = msvr(Xtrain, Ytrain, ker, C, epsi, par, tol)


# %%
# Predict with train set
K = kernelmatrix('rbf', Xtrain, Xtrain, par)
Ypred = np.dot(K, Beta)


# %%
# Predict with test set
K = kernelmatrix('rbf', Xtest, Xtrain, par)
Ypred = np.dot(K, Beta)


# %%



