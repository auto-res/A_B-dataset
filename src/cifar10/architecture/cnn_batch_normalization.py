import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)  # バッチ正規化を追加
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)  # バッチ正規化を追加
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)  # バッチ正規化を追加
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)  # バッチ正規化を追加
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 畳み込み -> バッチ正規化 -> 活性化関数 -> プーリング
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 畳み込み -> バッチ正規化 -> 活性化関数 -> プーリング
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.bn3(self.fc1(x)))  # 全結合 -> バッチ正規化 -> 活性化関数
        x = F.relu(self.bn4(self.fc2(x)))  # 全結合 -> バッチ正規化 -> 活性化関数
        x = self.fc3(x)  # 出力層
        return x

def train(trainloader, params):
    net = Net()

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
