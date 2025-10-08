# **Linear Regression**

## 1. <u>***NumPy Implementation***</u>

### ***Initial State***
```python
import numpy as np
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
w = 200
b = 100
```

###  ***Prediction***
```python
def predict(x, w, b):
    return x * w + b
```

### ***Compute Cost***
- **$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \big( w x^{(i)} + b - y^{(i)} \big)^2$**
```python
def compute_cost(x, y, w, b):
    return np.sum((w * x + b - y) ** 2) / (2 * x.shape[0])
```

### ***Compute Gradient***
- **$\frac{\partial J(w, b)}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \big( w x^{(i)} + b - y^{(i)} \big) x^{(i)}$**
</br>

- **$\frac{\partial J(w, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \big( w x^{(i)} + b - y^{(i)} \big)$**

```python
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    return np.sum((w * x + b - y) * x) / m, np.sum(w * x + b - y) / m
```

### ***Gradient Descent***
- **$w_j := w_j - \alpha \frac{\partial J(w, b)}{\partial w_j}$**
</br>

- **$b := b - \alpha \frac{\partial J(w, b)}{\partial b}$**

```python
def gradient_descent(x, y, w, b, alpha, num_iters, cost_function, gradient_function):
    for _ in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b
```

## 2. <u>***Scikit-Learn Implementation***</u>

### ***Import***
```python
from sklearn.linear_model import LinearRegression
```

### ***Create and train model***
```python
model = LinearRegression()
model.fit(x, y)
```

### ***Predictions***
```python
y_pred = model.predict(x)
```

### ***Parameters***
```python
w = model.coef_[0]
b = model.intercept_
```
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
- **$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \big( (\mathbf{w}^T \mathbf{x}^{(i)} + b) - y^{(i)} \big)^2$**
```python
def compute_cost(X, y, w, b):
  return np.sum(((np.dot(X, w) + b) - y) ** 2) * (1 / (2 * X.shape[0]))
```

### ***Compute gradient***
- **$ \frac{\partial J(w, b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \big( (\mathbf{w}^T \mathbf{x}^{(i)} + b) - y^{(i)} \big) x_j^{(i)} $**
</br>

- **$ \frac{\partial J(w, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \big( (\mathbf{w}^T \mathbf{x}^{(i)} + b) - y^{(i)} \big) $**

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
- **$w_j := w_j - \alpha \frac{\partial J(w, b)}{\partial w_j}$**
</br>

- **$b := b - \alpha \frac{\partial J(w, b)}{\partial b}$**

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
- **$X = \frac{X - \mu}{X_{\text{max}} - X_{\text{min}}}$**

```python
def mean_normalize_features(X):
    mu = np.mean(X, axis = 0)
    maxValue = np.max(X)
    minValue = np.min(X)
    X = (X - mu) / (maxValue - minValue)
``` 
### ***Feature Scaling: Z-Score Normalization***
- **$X= \frac{X - \mu}{\sigma}$**

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
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
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

## 1. <u>***NumPy Implementation***</u>

### ***Sigmoid Function***
- **$\sigma(z) = \frac{1}{1 + e^{-z}}$**
```python
def sigmoid(z):
    return 1/(1+np.exp(-z))
```

### ***Decision Boundary***
```python
# Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# Plot the decision boundary
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()
```



### ***Compute Cost***
- **$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(f_{w,b}^{(i)}) + (1 - y^{(i)}) \log(1 - f_{w,b}^{(i)}) \Big]$**
```python
def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]

    # Compute the model predictions for all examples
    z = np.dot(X, w) + b
    f_wb = 1 / (1 + np.exp(-z))   # sigmoid vectorized

    # Compute cost using vectorized operations
    cost = (-1 / m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    return cost
```

### ***Compute Gradient***
- **$\text{error}^{(i)} = \hat{y}^{(i)} - y^{(i)}$**
</br>

- **$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \text{error}^{(i)} \cdot X_{ij}, \quad j = 1, \dots, n$**
</br>

- **$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \text{error}^{(i)}$**

```python
def compute_gradient_logistic(X, y, w, b):
    m = X.shape[0]
    # Compute predictions for all examples at once
    f_wb = sigmoid(np.dot(X, w) + b)  # shape (m,)
    # Compute the error vector
    err = f_wb - y                     # shape (m,)
    # Vectorized gradient computation
    dj_dw = np.dot(X.T, err) / m       # shape (n,)
    dj_db = np.sum(err) / m             # scalar
    return dj_db, dj_dw
```

### ***Gradient Descent***

- **$w_j := w_j - \alpha \frac{\partial J(w, b)}{\partial w_j}$**
</br>

- **$b := b - \alpha \frac{\partial J(w, b)}{\partial b}$**

```python
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   
        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
    return w, b
```

## 2. <u>***Scikit-Learn Implementation***</u>
### ***Dataset***
```python
import numpy as np
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
```

### ***Fit the model***
```python
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X, y)
```

### ***Make Predictions***
```python
y_pred = lr_model.predict(X)
```

### ***Calculate Accuracy***
```python
print(lr_model.score(X,y))
```