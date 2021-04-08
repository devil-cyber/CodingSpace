---
layout: single
header:
  overlay_color: "#333"
  teaser: https://devil-cyber.github.io/CodingSpace/assets/images/algorithm/tf.png
title:  "Get Started With Tensorflow"
excerpt: "Deep Learning & Machine Learning"
breadcrumbs: true
share: true
permalink: /TensorFlow/
date:    2021-04-04
toc: false

---

# What is TensorFlow?
* **Open-source library for graph-based numerical computation**
* **Developed by the Google Brain Team**
* **Low and high level APIs**
 - Addition, multiplication, differentiation
 - Machine learning & deep learning models


# What is a tensor?
* Generalization of vectors and matrices
* Collection of numbers
* Specific shape

## Overview of Concepts

TensorFlow gets its name from **tensors**, which are arrays of arbitrary dimensionality. Using TensorFlow, you can manipulate tensors with a very high number of dimensions. That said, most of the time you will work with one or more of the following low-dimensional tensors:

  * A **scalar** is a 0-d array (a 0th-order tensor).  For example, `"Manikant"` or `5`
  * A **vector** is a 1-d array (a 1st-order tensor).  For example, `[2, 3, 5, 7, 11]` or `[5]`
  * A **matrix** is a 2-d array (a 2nd-order tensor).  For example, `[[3.1, 8.2, 5.9][4.3, -2.7, 6.5]]`

**Scaler with dim 0**
![](https://www.tensorflow.org/guide/images/tensor/scalar.png)
**vector with dim 1**
![](https://www.tensorflow.org/guide/images/tensor/vector.png)
**Matrix with dim 2**
![](https://www.tensorflow.org/guide/images/tensor/matrix.png)

# Defining tensors in TensorFlow


```python
import tensorflow as tf

# 0D Tensor
d0 = tf.ones((1,))
# 1D Tensor
d1 = tf.ones((2,))
# 2D Tensor
d2 = tf.ones((2, 2))
# 3D Tensor
d3 = tf.ones((2, 2, 2))
```

# Defining constants in TensorFlow
**A constant is the simplest category of tensor**
* Not trainable
* Can have any dimension


```python
from tensorflow import constant
# Define a 2x3 constant.
a = constant(3, shape=[2, 3])
# Define a 2x2 constant.
b = constant([1, 2, 3, 4], shape=[2, 2])
```

# Defining and initializing variables


```python
# Define a variable
a0 = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.float32)
a1 = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.int16)
```

# Applying the addition operator


```python
 from tensorflow import constant, add
 # Define 0-dimensional tensors
A0 = constant([1])
B0 = constant([2])
# Define 1-dimensional tensors
A1 = constant([1, 2])
B1 = constant([3, 4])
# Define 2-dimensional tensors
A2 = constant([[1, 2], [3, 4]])
B2 = constant([[5, 6], [7, 8]])

# Applying the addition operator
# Perform tensor addition with add()
C0 = add(A0, B0)
C1 = add(A1, B1)
C2 = add(A2, B2)
```

# How to perform multiplication in TensorFlow
* **Element-wise multiplication performed using multiply() operation**
- The tensors multiplied must have the same shape
- E.g. [1,2,3] and [3,4,5] or [1,2] and [3,4]
* **Matrix multiplication performed with matmul() operator**
- The matmul(A,B) operation multiplies A by B
- Number of columns of A must equal the number of rows of B


```python
# Import operators from tensorflow
from tensorflow import ones, matmul, multiply
# Define tensors
A0 = ones(1)
A31 = ones([3, 1])
A34 = ones([3, 4])
A43 = ones([4, 3])

# What types of operations are valid?
# multiply(A0, A0) , multiply(A31, A31) , and multiply(A34, A34)
# matmul(A43, A34 ), but not matmul(A43, A43)
```
# Overview of advanced operations
 
| Operations   |      Use    
|---------- |:-------------                                 
| gradient()|  Computes the slope of a function at a point  
| reshape()   |    Reshapes a tensor (e.g. 10x10 to100x1)                                
| random()   | Populates tensor with entries drawn from a probability distribution                               

# Finding the optimum
* In many problems, we will want to find the optimum of a function.
   - ## `Minimum: Lowest value of a loss function.`
   - ## `Maximum: Highest value of objective function.`
* We can do this using the gradient() operation.
   - ## `Optimum: Find a point where gradient = 0.`
   - ## `Minimum: Change in gradient > 0`
   - ## `Maximum: Change in gradient < 0`


```python
# Import tensorflow under the alias tf
import tensorflow as tf
# Define x
x = tf.Variable(-1.0)
```


```python
# Define y within instance of GradientTape
with tf.GradientTape() as tape:
  tape.watch(x)
  y = tf.multiply(x, x)
```


```python
# Evaluate the gradient of y at x = -1
g = tape.gradient(y, x)
print(g.numpy())
```

    -2.0


# How to reshape a grayscale image


```python
# Import tensorflow as alias tf
import tensorflow as tf
# Generate grayscale image
gray = tf.random.uniform([2, 2], maxval=255, dtype='int32')
# Reshape grayscale image
gray = tf.reshape(gray, [2*2, 1])
```

# How to reshape a color image


```python
# Import tensorflow as alias tf
import tensorflow as tf
# Generate color image
color = tf.random.uniform([2, 2, 3], maxval=255, dtype='int32')
# Reshape color image
color = tf.reshape(color, [2*2, 3])
```

# Introduction to loss functions
* Fundamental tensorflow operation
  - ## `Used to train a model`
  - ## `Measure of model fit`
* Higher value -> worse fit
   - ## `Minimize the loss function`

# Common loss functions in TensorFlow
* TensorFlow has operations for common loss functions
  - ## `Mean squared error (MSE)`
  - ## `Mean absolute error (MAE)`
  - ## `Huber error`
* Loss functions are accessible from tf.keras.losses()
  - ## `tf.keras.losses.mse()`
  - ## `tf.keras.losses.mae()`
  - ## `tf.keras.losses.Huber()`

# Why do we care about loss functions?
* MSE
  - Strongly penalizes outliers
  - High (gradient) sensitivity near minimum
* MAE
  - Scales linearly with size of error
  - Low sensitivity near minimum
* Huber
  - Similar to MSE near minimum
  - Similar to MAE away from minimum


```python
# Import TensorFlow under standard alias
import tensorflow as tf
# Compute the MSE loss
loss = tf.keras.losses.mse(targets, predictions)
```

