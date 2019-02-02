import numpy as np
"""
In this lesson we prepare simple feedforward neural network

"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(f'Working on device={device}')

# Hyper-parameters

# each CIFAR image is RGB 32x32, so it is an 3D array [3,32,32]
# we will flatten the image as vector dim=3*32*32
input_size = 3*32*32

hidden_size = 512
# we have 10 classes

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

num_classes = 10

num_epochs = 5
batch_size = 16

learning_rate = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False)


class FeedForwardNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        '''
        Fully connected neural network with one hidden layer
        '''
        super(FeedForwardNeuralNet, self).__init__()

        # first hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # activation function
        self.relu = nn.ReLU()
        # output layer
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        '''
        This method takes an input x and layer after layer compute network states.
        Last layer gives us predictions.
        '''
        state = self.fc1(x)
        state = self.relu(state)
        state = self.output(state)
        #out = self.softmax(out)
        return state


model = FeedForwardNeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss function,
criterion = nn.CrossEntropyLoss()

# Optimizer define how we will chanage model weights
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model - training loop

# set our model in the training mode
model.train()
for epoch in range(num_epochs):

    epoch_loss = 0
    # data loop, iterate over chunk of data(batch) eg. 32 elements
    # compute model prediction
    # update weights
    for i, batch_sample in enumerate(train_loader):

        # print(batch_sample)
        images, labels = batch_sample

        # flatten the image and move to device
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        # Forward pass, compute prediction,
        # method 'forward' is automatically called
        prediction = model(images)
        # Compute loss, quantify how wrong our predictions are
        # small loss means a small error
        loss = criterion(prediction, labels)
        epoch_loss += loss.item()

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / len(train_loader)

    # Test the model

    # set our model in the training mode
    model.eval()
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            # reshape image
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            # predict classes
            prediction = model(images)

            # compute accuracy
            _, predicted = torch.max(prediction.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct/total

        # Accuracy of the network on the 10000 test images
        print(f'Epoch [{epoch+1}/{num_epochs}]], Loss: {epoch_loss:.4f} Test acc: {acc}')
