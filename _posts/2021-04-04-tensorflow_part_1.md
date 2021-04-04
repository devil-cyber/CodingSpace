---
layout: single
header:
  overlay_color: "#333"
  teaser: /assets/images/algorithm/tf.png
title:  "Get Started With Basic Of Tensorflow"
excerpt: "Deep Learning & Machine Learning"
breadcrumbs: true
share: true
permalink: /tensorflow_part_1/
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
* Specfic shape

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

