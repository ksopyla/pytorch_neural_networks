# Pytorch neural networks tutorial

This repository contains easy to follow Pytorch tutorial for beginners and intermediate students. The goal is to  introduce you to pytorch on practical examples. I hope that it will help you to start your journey with neural networks.

There are few popular neural network architecture which I teach on workshops or bootcamps like: feedforward, convolutional and recurrent.

In this tutorial we build:
* Feedforward neural network with one hidden layer for CIFAR-10 classification
* Feedforward neural network with three hidden layers for CIFAR-10 classification
* Convolutional neural network for CIFAR-10 classification
* LSTM recurrent neural network for counting chars in long text
* LSTM recurrent neural network for IMDB sentiment analisys with torchtext

## Prerequistis

This tutorial was written and tested on Ubuntu 18.10, 

* Python - version >= 3.6 
* pipenv - package and virtual environment management 
* numpy
* matplotlib
* pytorch - version >= 1.0
* torchtext - version >= 0.4


1. Install Python.
1. Install pipenv - https://pipenv.readthedocs.io/en/latest/install/#pragmatic-installation-of-pipenv
1. Git clone the repository
1. Install all necessary python packages executing this command in terminal

```
cd python_neural_network
pipenv install
```


## Repository structure

* ./data - folder for downloaded dataset, all data we are working with are automatically downloaded at first use


## Feedforward neural network for classifying CIFAR-10

Network for classifying [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) images into one of 10 categories: ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

### Feedforward neural network with one hidden layer
We build simple network with 1 hidden layer and output layer. As input we pass raw image pixels as 32x32 vector of numbers.   

File: **[feedforward_1_hid_nn.py](https://github.com/ksopyla/pytorch_neural_networks/blob/master/feedforward_1_hid_nn.py)**
This model achieve ~ 48% accuracy after 5 epoch.

Sample output
```
Epoch [1/5]], Loss: 1.7713 Test acc: 0.4423
Epoch [2/5]], Loss: 1.6124 Test acc: 0.4582
Epoch [3/5]], Loss: 1.5280 Test acc: 0.466
Epoch [4/5]], Loss: 1.4640 Test acc: 0.4889
Epoch [5/5]], Loss: 1.4130 Test acc: 0.4799
```

### Feedforward neural network with three hidden layers
Analogous to previous model feedforward network with 3 hidden layers and output layer. As input we pass raw image pixels as 32x32 vector of numbers.   

File: **[feedforward_3_hid_nn.py](https://github.com/ksopyla/pytorch_neural_networks/blob/master/feedforward_3_hid_nn.py)**
This model achieve ~ 51% accuracy after 5 epoch.

Sample output
```
Epoch [1/5]], Loss: 1.6971 Test acc: 0.4627
Epoch [2/5]], Loss: 1.4874 Test acc: 0.4947
Epoch [3/5]], Loss: 1.3887 Test acc: 0.493
Epoch [4/5]], Loss: 1.3108 Test acc: 0.5144
Epoch [5/5]], Loss: 1.2406 Test acc: 0.5166
```

## Convolutional neural network 

This model uses convolutional neural network with 3 convolution layers and output layer. As input we pass raw image pixels as 32x32 vector of numbers.   

File: **[conv_net_cifar.py](https://github.com/ksopyla/pytorch_neural_networks/blob/master/conv_net_cifar.py)**
This model achieve ~ 51% accuracy after 5 epoch.

Sample output
```
Epoch [1/5]], Loss: 1.6971 Test acc: 0.4627
Epoch [2/5]], Loss: 1.4874 Test acc: 0.4947
Epoch [3/5]], Loss: 1.3887 Test acc: 0.493
Epoch [4/5]], Loss: 1.3108 Test acc: 0.5144
Epoch [5/5]], Loss: 1.2406 Test acc: 0.5166
```
## Recurrent neural network 

TODO: