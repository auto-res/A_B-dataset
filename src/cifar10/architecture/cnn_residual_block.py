import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class Net(nn.Module):
    def __init__(self, block):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.layer1 = self.make_layer(block, 16, 16, 2)
        self.layer2 = self.make_layer(block, 16, 32, 2)
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 32 * 5 * 5)
        x = self.fc(x)
        return x

def train(trainloader, params, device):
    net = Net()
    net.to(device)  # Move the model to the specified device

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=params['lr'])

    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)  # Move inputs and labels to the specified device

            optimizer.zero_grad()  # zero the parameter gradients

            outputs = net(inputs)  # forward + backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                running_loss = 0.0

    return net

def test(net, testloader, device):
    all_outputs = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  # Move images to the specified device
            outputs = net(images)
            probabilities = F.softmax(outputs, dim=1)
            probabilities_np = probabilities.cpu().numpy().tolist()  # Move the probabilities back to CPU for numpy conversion
            all_outputs.extend(probabilities_np)

    all_outputs_np = np.array(all_outputs)
    return all_outputs_np

def model(trainloader, testloader, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = train(trainloader, params, device)
    y_pred = test(net, testloader, device)
    return y_pred
