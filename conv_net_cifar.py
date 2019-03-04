import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

# check if cuda is enabled
USE_GPU=1
# Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')


# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(f'Using {device} device')

num_classes = 10
num_epochs = 5
batch_size = 4


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# plt.show()

# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        # channel_in=3 channels_out=8
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1)

        # 24 chaneels by 8x8 pixesl
        self.fc1 = nn.Linear(24 * 8 * 8, 100)

        self.output = nn.Linear(100, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        # max_pooling will resize input from 32 to 16
        x = self.pool(F.relu(self.conv2(x)))
        # max_pooling will resize input from 16 to 8
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 24 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


net = ConvNet()
net.to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)


for epoch in range(num_epochs):  # loop over the dataset multiple times

    start_time = datetime.now()
    net.train()
    running_loss = 0.0
    epoch_loss = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        # move data to device (GPU if enabled, else CPU do nothing)
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(trainloader)

    time_elapsed = datetime.now() - start_time

    # Test the model

    # set our model in the training mode
    net.eval()
    # In test phase, we don't need to compute gradients (for memory efficiency)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Accuracy of the network on the 10000 test images
    acc = correct/total
    print(
        f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f} Test acc: {acc} time={time_elapsed}')


print('Finished Training')

# Detailed accuracy per class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
