# Multiple Support Vector Regression

Multiple support vector regression is a method which implements support vector regression with multi-input and multi-output. This package is based on the paper, [Multi-dimensional function approximation and regression estimation](https://link.springer.com/chapter/10.1007/3-540-46084-5_123), [F Pérez-Cruz](https://scholar.google.com/citations?user=8FfrHw0AAAAJ&hl=en&oi=sra).

## Usage

```python
from msvr import kernelmatrix
from msvr import msvr
import numpy as np

# Construct x samples (input) and y samples (output)
# x: num_samples * dimension
# y: num_smaples * dimension
x1 = np.sin(np.arange(0, 9, 0.01))
x2 = np.cos(np.arange(0, 9, 0.01))
x3 = x1**2
x4 = (x1+x2)/2

x = np.vstack((x1,x2)).T
y = np.vstack((x3,x4)).T

# Input & Output
# Xtrain: number of samples * input dimension
# Ytrain: number of samples * output dimension

Xtrain = x[:600, :]
Ytrain = y[:600, :]
Xtest = x[600:, :]
Ytest = y[600:, :]
Xtrain = (Xtrain-np.min(Xtrain))/(np.max(Xtrain)-np.min(Xtrain))
Ytrain = (Ytrain-np.min(Ytrain))/(np.max(Ytrain)-np.min(Ytrain))

'''
Parameters
  ker: kernel ('lin', 'poly', 'rbf'),
  C: cost parameter,
  par (kernel):
	-lin: no parameters,
	-poly: [gamma, b, degree],
	-rbf: sigma (width of the RBF kernel),
  tol: tolerance.
'''

ker     = 'rbf'
C       = 2
epsi = 0.001
par   = 0.8 # if kernel is 'rbf', par means sigma
tol     = 1e-10

# Train
Beta = msvr(Xtrain, Ytrain, ker, C, epsi, sigma, tol)

# Predict with train set
H = kernelmatrix('rbf', Xtrain, Xtrain, sigma);
Ypred = np.dot(H, Beta)

# Predict with test set
H = kernelmatrix('rbf', Xtest, Xtrain, sigma);
Ypred = np.dot(H, Beta)
```

## Kernel function

<img src="https://github.com/KaishuaiXu/msvr/blob/master/pic/kernel.png?raw=true" alt="kernel function" width="395" height="175" />

