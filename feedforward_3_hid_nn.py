"""
In this lesson we prepare feedforward neural network with 3 hidden layers

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

# each cifar image is RGB 32x32, so it is an 3D array [3,32,32]
# we will flatten the image as vector dim=3*32*32 
input_size = 3*32*32

hidden_size = 512
# we have 10 classes

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

num_classes = 10

num_epochs = 2
batch_size = 8

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


#import matplotlib.pyplot as plt
import numpy as np


class MultilayerNeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        '''
        Fully connected neural network with 3 hidden layers
        '''
        super(MultilayerNeuralNet, self).__init__()
        
        # hidden layers sizes, you can play with it as you wish!
        hidden1 = 512
        hidden2 = 256
        hidden3 = 128

        # input to first hidden layer parameters
        self.fc1 = nn.Linear(input_size, hidden1) 
        self.relu1 = nn.ReLU()

        # second hidden layer
        self.fc2 = nn.Linear(hidden1, hidden2) 
        self.relu2 = nn.ReLU()
        
        # third hidden layer
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.relu3 = nn.ReLU()

        # last output layer
        self.output = nn.Linear(hidden3, num_classes) 

    
    def forward(self, x):
        '''
        This method takes an input x and layer after layer compute network states.
        Last layer gives us predictions.
        '''
        state = self.fc1(x)
        state = self.relu1(state)

        state = self.fc2(x)
        state = self.relu2(state)

        state = self.fc3(x)
        state = self.relu3(state)

        state = self.output(state)
        
        return state

model = MultilayerNeuralNet(input_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, batch_sample in enumerate(train_loader):  

        #print(batch_sample)
        images, labels = batch_sample
        
        # Move tensors to the configured device
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 1000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')