# Pytorch neural networks tutorial

This repository contains easy to follow Pytorch tutorial for beginners and intermediate students. The goal is to introduce you to Pytorch on practical examples. I hope that it will help you to start your journey with neural networks.

There are a few popular neural network architecture which I teach on workshops or boot camps like feedforward, convolutional, recurrent, transformer. Examples will help you with image and text classification tasks.

Image classification:
* Single layer fully connected neural network for CIFAR-10 classification. 
* Feedforward neural network with three hidden layers for CIFAR-10 classification with ReLu activation.
* [Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) for CIFAR-10 with 3 convolution layer with and fully connected output layer, as activation we use ReLu.


NLP tasks:
* LSTM recurrent neural network for counting chars in a long text.
* LSTM recurrent neural network for IMDB sentiment analysis with truncated backpropagation through time (LSTM TBTT)



TODO: 
* (todo) LSTM recurrent neural network for multilabel classification (Toxicity dataset)
* (todo) Transformer for text classification on IMDB

## Prerequisites

This tutorial was written and tested on Ubuntu 18.10, 

* Python - version >= 3.6 
* pipenv - package and virtual environment management 
* numpy
* matplotlib
* pytorch - version >= 1.0
* torchtext - version >= 0.4


1. Install Python.
1. [Install pipenv](https://pipenv.readthedocs.io/en/latest/install/#pragmatic-installation-of-pipenv)
1. Git clone the repository
1. Install all necessary python packages executing this command in terminal

```
git clone https://github.com/ksopyla/pytorch_neural_networks.git
cd pytorch_neural_networks
pipenv install
```

## Repository structure

* ./data - folder for a downloaded dataset, all data we are working with are automatically downloaded at first use


## Feedforward neural network for classifying CIFAR-10

Network for classifying [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) images into one of 10 categories: ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

It is worth checking current SotA results on CIFAR-10
* [Who is the best in CIFAR-10?](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130)


### Feedforward neural network with one hidden layer
We build a simple network with 1 hidden layer and an output layer. As input, we pass raw image pixels as the 32x32 vector of numbers.   

File: **[feedforward_1_hid_nn.py](feedforward_1_hid_nn.py)**
This model achieve ~ 48% accuracy after 5 epoch.

Model summary:
* input layer: 3x32x32 (3 rgb channels times image resolution 32pixels)
* hidden layer: 512 neurons
* output layer: 10 neurons, each reflects probability of belonging to class 

Sample output
```
Epoch [1/5]], Loss: 1.7713 Test acc: 0.4423
Epoch [2/5]], Loss: 1.6124 Test acc: 0.4582
Epoch [3/5]], Loss: 1.5280 Test acc: 0.466
Epoch [4/5]], Loss: 1.4640 Test acc: 0.4889
Epoch [5/5]], Loss: 1.4130 Test acc: 0.4799
```

### Feedforward neural network with three hidden layers
Analogous to previous model feedforward network with 3 hidden layers and output layer. 

This is an upgraded version of the previous model, between input and output we added 3 fully connected hidden layers. Adding more layers makes the network more expressive but harder to train. The three new problems could emerge vanishing gradients, model overfitting, and computation time complexity. In our case where the dataset is rather small, we did not see those problems in real scale.


File: **[feedforward_3_hid_nn.py](feedforward_3_hid_nn.py)**
This model achieve ~ 51% accuracy after 5 epoch.

Model summary:
* input layer: 3x32x32 (3 rgb channels times image resolution 32pixels)
* 3 x hidden layers: 512, 256, 128 neurons
* output layer: 10 neurons, each reflects probability of belonging to class 


Sample output
```
Epoch [1/5]], Loss: 1.6971 Test acc: 0.4627
Epoch [2/5]], Loss: 1.4874 Test acc: 0.4947
Epoch [3/5]], Loss: 1.3887 Test acc: 0.493
Epoch [4/5]], Loss: 1.3108 Test acc: 0.5144
Epoch [5/5]], Loss: 1.2406 Test acc: 0.5166
```

## Convolutional neural network for classifying CIFAR-10

This model uses a convolutional neural network with 3 convolution layers and an output layer. As input, we pass raw image pixels as the 32x32 vector of numbers.   

File: **[conv_net_cifar.py](conv_net_cifar.py)**
This model achieve ~ 67% accuracy after 5 epoch.

Model summary:
* conv 1 layer - kernel size 3x3, channels in 3, channels out 8,stride=1, padding=1
* relu - activation
* conv 2 layer - kernel size 3x3, channels in 8, channels out 16,stride=1, padding=1
* relu - activation
* max pooling - window size 2x2, it will downsize image 2x (from 32x32 to 16x16)
* conv 3 layer - kernel size 3x3, channels in 16, channels out 24,stride=1, padding=1
* relu - activation
* max pooling- window size 2x2, it will downsize image 2x (from 16x16 to 8x8)
* fully connected - 24x8x8x100
* fully connected(output) - 100x10


Sample output
```
Epoch [1/5]], Loss: 1.3465 Test acc: 0.6038 time=0:00:42.301045
Epoch [2/5]], Loss: 1.0122 Test acc: 0.64 time=0:00:42.168382
Epoch [3/5]], Loss: 0.8989 Test acc: 0.6649 time=0:00:41.995531
Epoch [4/5]], Loss: 0.8214 Test acc: 0.6834 time=0:00:42.099388
Epoch [5/5]], Loss: 0.7627 Test acc: 0.6761 time=0:00:42.047874
Finished Training
Accuracy of plane : 76 %
Accuracy of   car : 73 %
Accuracy of  bird : 42 %
Accuracy of   cat : 60 %
Accuracy of  deer : 63 %
Accuracy of   dog : 53 %
Accuracy of  frog : 72 %
Accuracy of horse : 69 %
Accuracy of  ship : 79 %
Accuracy of truck : 85 %

```


## LSTM recurrent neural network for counting chars in long text

In this example, we build the LSTM network which will work on text. Our goal is counting chars in text and predicting the most frequent one. Based on the provided code you will be able to adapt to almost any text classification task.
The code shows you how to process input text with TorchText, build and train recurrent n-layer LSTM with word embeddings.

File: **[lstm_net_counting_chars.py](RNN/lstm_net_counting_chars.py)**

This model achieves ~ 0.88 accuracy after 60 epoch.

Sample output
```
Epoch 0/60 loss=2.293760901405698 acc=0.19711539149284363 time=0:00:01.084109
Epoch 1/60 loss=2.1805509555907476 acc=0.125 time=0:00:01.066103
Epoch 2/60 loss=2.13575065514398 acc=0.11057692766189575 time=0:00:01.034095
...

Epoch 56/60 loss=0.017137109050675045 acc=0.9134615659713745 time=0:00:01.045637
Epoch 57/60 loss=0.03904954261249966 acc=0.8605769276618958 time=0:00:01.058192
Epoch 58/60 loss=0.031670229065985905 acc=0.8990384936332703 time=0:00:01.096598
Epoch 59/60 loss=0.022030536144498795 acc=0.889423131942749 time=0:00:01.144795

```

## LSTM recurrent neural network for IMDB movie review sentiment analysis

We build the LSTM network which will work on IMDB movie review text.

The code shows you how to process input text with TorchText, build and train recurrent 1-layer LSTM.

File: **[lstm_imdb.py](RNN/lstm_imdb.py)**

This model achieves ~ 0.87 accuracy after 10 epoch.

Sample output

```
Training Epoch 0/10 |################################| 391/391
Validation Epoch 0/10 |################################| 391/391
Epoch 0/10 loss=0.6549588166691763 acc=0.7202125787734985 time=0:10:02.022493
Training Epoch 1/10 |################################| 391/391
Validation Epoch 1/10 |################################| 391/391
Epoch 1/10 loss=0.594817126422282 acc=0.6989369988441467 time=0:06:25.937583
Training Epoch 2/10 |################################| 391/391
Validation Epoch 2/10 |################################| 391/391
Epoch 2/10 loss=0.5306539740556341 acc=0.7531969547271729 time=0:05:33.698079
Training Epoch 3/10 |################################| 391/391
Validation Epoch 3/10 |################################| 391/391
Epoch 3/10 loss=0.439730473110438 acc=0.7545236349105835 time=0:03:51.911136
Training Epoch 4/10 |################################| 391/391
Validation Epoch 4/10 |################################| 391/391
Epoch 4/10 loss=0.4551559637117264 acc=0.8277093768119812 time=0:04:08.334979
Training Epoch 5/10 |################################| 391/391
Validation Epoch 5/10 |################################| 391/391
Epoch 5/10 loss=0.3437549231759727 acc=0.8208919167518616 time=0:03:45.663289
Training Epoch 6/10 |################################| 391/391
Validation Epoch 6/10 |################################| 391/391
Epoch 6/10 loss=0.2964414275248947 acc=0.8551790118217468 time=0:03:45.582308
Training Epoch 7/10 |################################| 391/391
Validation Epoch 7/10 |################################| 391/391
Epoch 7/10 loss=0.30815188632444346 acc=0.8564338684082031 time=0:04:18.328589
Training Epoch 8/10 |################################| 391/391
Validation Epoch 8/10 |################################| 391/391
Epoch 8/10 loss=0.2521923514430785 acc=0.8667998909950256 time=0:03:44.280257
Training Epoch 9/10 |################################| 391/391
Validation Epoch 9/10 |################################| 391/391
Epoch 9/10 loss=0.2135891618821627 acc=0.8684223294258118 time=0:03:40.475379
```


## LSTM with TBTT recurrent neural network for IMDB movie review sentiment analysis

We build the LSTM network which will work on IMDB movie review text. This time we want to classify long text and show how to train recurrent network with use [Truncated Backpropagation through Time](https://machinelearningmastery.com/gentle-introduction-backpropagation-time/). This technique helps deal with vanishing and exploding gradients with very long sequences (long text, long amino sequence, time series). We will split gradient chain and do backpropagation every K-steps backward.

This example was coded based on suggestions from Pytorch forum threads:

* [Implementing Truncated Backpropagation Through Time
](https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500)
* [Correct way to do backpropagation through time?](https://discuss.pytorch.org/t/correct-way-to-do-backpropagation-through-time/11701)



The code shows you how to process input text with TorchText, build and train recurrent n-layer LSTM with pre-trained word embeddings.

File: **[lstm_imdb_tbptt.py](RNN/lstm_imdb_tbptt.py)**

This model achieves ~ 0.85 accuracy after 10 epoch.

Sample output
```
Training Epoch 0/10 |################################| 782/782
Validation Epoch 0/10 |################################| 782/782
Epoch 0/10 loss=3.116480209295402 acc=0.8292838931083679 time=0:01:19.544885
Training Epoch 1/10 |################################| 782/782
Validation Epoch 1/10 |################################| 782/782
Epoch 1/10 loss=1.9649102331837043 acc=0.8706841468811035 time=0:01:18.703602
Training Epoch 2/10 |################################| 782/782
Validation Epoch 2/10 |################################| 782/782
Epoch 2/10 loss=1.2844358206490802 acc=0.8699648380279541 time=0:01:18.822965
Training Epoch 3/10 |################################| 782/782
Validation Epoch 3/10 |################################| 782/782
Epoch 3/10 loss=0.7612629080134089 acc=0.8631713390350342 time=0:01:18.438742
Training Epoch 4/10 |################################| 782/782
Validation Epoch 4/10 |################################| 782/782
Epoch 4/10 loss=0.46042709653039493 acc=0.8654091954231262 time=0:01:18.089986
Training Epoch 5/10 |################################| 782/782
Validation Epoch 5/10 |################################| 782/782
Epoch 5/10 loss=0.3314593220629808 acc=0.8596547245979309 time=0:01:18.294183
Training Epoch 6/10 |################################| 782/782
Validation Epoch 6/10 |################################| 782/782
Epoch 6/10 loss=0.2812261906621592 acc=0.8589354157447815 time=0:01:18.187062
Training Epoch 7/10 |################################| 782/782
Validation Epoch 7/10 |################################| 782/782
Epoch 7/10 loss=0.2437611708150762 acc=0.8552589416503906 time=0:01:17.948963
Training Epoch 8/10 |################################| 782/782
Validation Epoch 8/10 |################################| 782/782
Epoch 8/10 loss=0.2500312502574547 acc=0.8591752052307129 time=0:01:18.136995
Training Epoch 9/10 |################################| 782/782
Validation Epoch 9/10 |################################| 782/782
Epoch 9/10 loss=0.2074765177977169 acc=0.8543797731399536 time=0:01:18.278987
```



## References and further reading

* [What is the difference between a Fully-Connected and Convolutional Neural Network?](https://www.reddit.com/r/MachineLearning/comments/3yy7ko/what_is_the_difference_between_a_fullyconnected/)
* [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)
* [Awesome Pytorch](https://github.com/bharathgs/Awesome-pytorch-list) - A curated list of dedicated pytorch resources. 
* [Pytorch tutorial](https://github.com/yunjey/pytorch-tutorial) - this repository provides tutorial code for deep learning researchers to learn PyTorch
* [Basics of Image Classification with PyTorch](https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864)