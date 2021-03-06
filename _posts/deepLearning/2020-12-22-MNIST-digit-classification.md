---
layout: post
title:  "MNIST Handwritten Digit Recognition using PyTorch"
date:   2020-12-22 18:41:29 +0530
categories: Deep Learning
permalink: /pytorch-neural-net/
---
## Let's get familiar with PyTorch and MNIST dataset
- [MNIST](http://yann.lecun.com/exdb/mnist/) database of handwritten digits,has a training set of 60,000 examples, and a test set of 10,000 examples.

<img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" width="340"/>

- [PyTorch](https://pytorch.org/) is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab. It is free and open-source software.

- For PyTorch tutorial follow this [Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)


## Let's get started with PyTorch and Neural Net

- Import the necessary library from PyTorch

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
```

- Device configuration for GPU or CPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

- Setup the required parameter

```python
input_size = 784 # 28 X 28 image in MNIST dataset
hidden_size = 100
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.001
```

- Download MNIST dataset

```python
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor())
```

- DataLoader class to load data using PyTorch

```python
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
example = iter(train_loader)
samples,label = example.next()
print(samples.shape,label.shape)
```
- Plot of dataset

```python
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.imshow(samples[i][0],cmap='gray')
plt.show()
```

- Neural net class

```python
class NeuralNet(nn.Module):
  def __init__(self,input_size,hidden_size,num_classes):
    super(NeuralNet,self).__init__()
    self.l1 = nn.Linear(input_size,hidden_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size,num_classes)
  def forward(self,x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    return out
```
- Load the model and optimizer

```python
model = NeuralNet(input_size,hidden_size,num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
```

- Let's start the training loop
```python
# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
  for i,(images,label) in enumerate(train_loader):
    # 100, 1,28, 28
    # 100, 784
    images = images.reshape(-1,28*28).to(device)
    labels = label.to(device)

    # forward 
    outputs = model(images)
    loss = criterion(outputs, labels)

    # backward
    optim.zero_grad()
    loss.backward()
    optim.step()

    if(i+1)%100 == 0:
      print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
```

- Let's test our model 
```python
# test loop
with torch.no_grad():
  n_correct = 0
  n_sample = 0
  for images,labels in test_loader:
    images = images.reshape(-1,28*28).to(device)
    labels = labels.to(device)
    outputs = model(images)
    _,predictions = torch.max(outputs,1)
    n_sample += labels.shape[0]
    n_correct += (predictions == labels).sum().item()
  print(n_correct)
  acc = 100 * n_correct / n_sample
  print(acc)
```

`Get the collab code` [here](https://colab.research.google.com/drive/1J7d2pNySIy3i55-3qgRLcSbbm7V2Y7Yh?usp=sharing)

> Happy Coding