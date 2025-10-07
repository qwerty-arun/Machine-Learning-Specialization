# **Linear Regression**

---

# **Multiple Linear Regression**

## 1. <u>***NumPy Implementation***</u>

### ***Intial State***

```python
import numpy as np
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
```

### ***Predict single for single example***

```python
def predict(x, w, b):
  return np.dot(x, w) + b
```

### ***Compute Cost***

```python
def compute_cost(X, y, w, b):
  return np.sum(((np.dot(X, w) + b) - y) ** 2) * (1 / (2 * X.shape[0]))
```

### ***Compute gradient***

```python
def compute_gradient(X, y, w, b):
    m = X.shape[0]  # number of examples

    predictions = np.dot(X, w) + b  # shape (m,)
    error = predictions - y  # shape (m,)
    # Compute gradients
    dj_dw = (1 / m) * np.dot(X.T, error)  # shape (n,)
    dj_db = (1 / m) * np.sum(error)       # scalar

    return dj_db, dj_dw
```

### ***Gradient Descent***
```python
import copy

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    w = copy.deepcopy(w_in)  # avoid modifying global w
    b = b_in

    for _ in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b
```
### ***Feature Scaling: Mean Normalization***
```python
def mean_normalize_features(X):
    mu = np.mean(X, axis = 0)
    maxValue = np.max(X)
    minValue = np.min(X)
    X = (X - mu) / (maxValue - minValue)

``` 
### ***Feature Scaling: Z-Score Normalization***
```python
def zscore_normalize_features(X):
    mu     = np.mean(X, axis=0)
    sigma  = np.std(X, axis=0)
    X_norm = (X - mu) / sigma      
    return (X_norm, mu, sigma)
```

## 2. <u>***Scikit-Learn Implementation***</u>

### ***Import***

```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
```


### ***Scale/normalize data***

```python
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
```


### ***Create and fit regression model***

```python
sgdr = SDGRegressor(max_iter=1000)
sdgr.fit(X_norm, y_train)
```

### ***Parameters***

```python
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
```

### ***Predictions***
```python
y_pred_sgd = sgdr.predict(X_norm)
y_pred = np.dot(X_norm, w_norm) + b_norm
```
---
# ***Logistic Regression***
