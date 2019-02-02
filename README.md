# Pytorch neural networks tutorial

This repository contains some Pytorch tutorial for beginners. The goal is to gently introduce you to pytorch on practical examples. I hope that it will help you start your journey with neural networks.

There are few popular neural network architecture which I teach on workshops or bootcamps like: feedforward, convolutional and recurrent.

## Prerequistis

This tutorial was written and tested on Ubuntu 16.10, 

* Python 3.6 or above
* pipenv - package and virtual environment management 
* numpy
* matplotlib
* pytorch (torch, torchvision, torchtext) min version. 1.0


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
Epoch [1/20]], Loss: 1.7711 Test acc: 0.4423
Epoch [2/20]], Loss: 1.6173 Test acc: 0.4663
Epoch [3/20]], Loss: 1.5349 Test acc: 0.4694
Epoch [4/20]], Loss: 1.4676 Test acc: 0.4939
Epoch [5/20]], Loss: 1.4155 Test acc: 0.4846
```

### Feedforward neural network with three hidden layers
Analogous to previous model feedforward network with 3 hidden layers and output layer. As input we pass raw image pixels as 32x32 vector of numbers.   

File: **[feedforward_3_hid_nn.py](https://github.com/ksopyla/pytorch_neural_networks/blob/master/feedforward_3_hid_nn.py)**
This model achieve ~ 47% accuracy.



## Convolutional neural network 



## Recurrent neural network 