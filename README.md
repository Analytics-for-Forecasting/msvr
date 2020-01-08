# Multiple Support Vector Regression

Multiple support vector regression is a method which implements support vector regression with multi-input and multi-output. This package is based on the paper, [Multi-dimensional function approximation and regression estimation](https://link.springer.com/chapter/10.1007/3-540-46084-5_123), [F Pérez-Cruz](https://scholar.google.com/citations?user=8FfrHw0AAAAJ&hl=en&oi=sra).

## Requirement
numpy
sklearn

## Usage

```python
from model.MSVR import MSVR
from model.utility import create_dataset, rmse
import numpy as np

# Construct x samples (input) and y samples (output)
# x: num_samples * inputDim
# y: num_smaples * outputH
ts = np.sin(np.arange(0, 9, 0.01)).reshape(-1)

segmentation = int(len(ts)*2/3)
dim = 50
h = 5

dataset = create_dataset(ts, dim, h)
X, Y = dataset[:, :(0 - h)], dataset[:, (0-h):]
train_input = X[:segmentation, :]
train_target = Y[:segmentation].reshape(-1, h)
test_input = X[segmentation:, :]
test_target = Y[segmentation:].reshape(-1, h)

msvr = MSVR(kernel = 'rbf', gamma = 0.1, epsilon=0.001)
# Train
msvr.fit(train_input, train_target)

# Predict with train set
trainPred = msvr.predict(train_input)
# Predict with test set
testPred = msvr.predict(test_input)

trainMetric = rmse(train_target,trainPred)
testMetric = rmse(test_target,testPred)

print(trainMetric, testMetric)
```

## Kernels

This module implements [sklearn.metrics.pairwise.pairwise_kernels](https://scikit-learn.org/stable/modules/metrics.html#metrics) to support multiple kernels. A brief example is given there:
```
msvr = MSVR(kernel = 'rbf', gamma = 0.1)
```
The valid metric for kernels, and the kernel functions the map to, are:
| Metric | Function |
| --- | --- |
| 'additive_chi2' | sklearn.pairwise.additive_chi2_kernel  |
| 'chi2'          | sklearn.pairwise.chi2_kernel           |
| 'linear'        | sklearn.pairwise.linear_kernel         |
| 'poly'          | sklearn.pairwise.polynomial_kernel     |
| 'polynomial'    | sklearn.pairwise.polynomial_kernel     |
| 'rbf'           | sklearn.pairwise.rbf_kernel            |
| 'laplacian'     | sklearn.pairwise.laplacian_kernel      |
| 'sigmoid'       | sklearn.pairwise.sigmoid_kernel        |
| 'cosine'        | sklearn.pairwise.cosine_similarity     |

