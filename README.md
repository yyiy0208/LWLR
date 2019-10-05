# LWLR
Hw. 5 Locally Weighted Linear Regression

## Locally Weighted Linear Regression
When exploring the relationship between non-linear objects, using traditional linear regression model may lead to under-fitting.
By introducing locally weighted linear regression which gives higher weights to samples closer to the object we want to predict, 
we can improve the performance of linear model.

## Dependence
Locally weighted linear regression is a modified version of `sklearn.linear_model.LinearRegression`. `numpy` is used to do vector operations.
- [numpy](https://numpy.org/)
- [sklearn](https://scikit-learn.org/)


## Quick Start

**Import the module**
```python
from LWLR import LWLR
```

**Initialize model**
```python
model = LWLR(k=0.1)

```
k indicates the weights we give to the neighbors of the object we want to predict. 
The smaller the value of k, the larger weight we give. 
The default value of k is set 0.1. 


**Train**
```python
model.fit(train,target)
```
All inputs should be numpy arrays. `train` should be 2D array and `target` should be 1D array.

**Predict**
```python
model.predict(test)
```
`test` should be 2D numpy array.
Return predictions as numpy array.
