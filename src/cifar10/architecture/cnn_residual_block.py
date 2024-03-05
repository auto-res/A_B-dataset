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

def train(trainloader, params):
    net = Net(ResidualBlock)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=params['lr'], momentum=0.9)

    for epoch in range(1):  # データセットを複数回繰り返して学習
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 入力データを取得
            inputs, labels = data

            # 勾配をゼロにする
            optimizer.zero_grad()

            # 順伝播 + 逆伝播 + 最適化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 統計を表示
            running_loss += loss.item()
            if i % 2000 == 1999:    # 2000ミニバッチごとに表示
                #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    return net

def test(net, testloader):
    all_outputs = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            probabilities = F.softmax(outputs, dim=1)
            probabilities_np = probabilities.numpy().tolist()
            all_outputs.extend(probabilities_np)

    all_outputs_np = np.array(all_outputs)
    return all_outputs_np

def model(trainloader, testloader, params):
    net = train(trainloader, params)
    y_pred = test(net, testloader)
    return y_pred
