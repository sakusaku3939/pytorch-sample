import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # input: 3チャンネル(RGB), output: 21チャンネル
        self.conv1 = nn.Conv2d(3, 21, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 21チャンネル, output: 40チャンネル
        self.conv2 = nn.Conv2d(21, 40, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 出力サイズ = (入力サイズ - kernel_size + 2 * padding) / stride + 1
        # 1. conv1の出力サイズ: (224 - 5 + 2 * 0) / 1 + 1 = 220
        # 2. pool1の出力サイズ: (220 - 2 + 2 * 0) / 2 + 1 = 110
        # 3. conv2の出力サイズ: (110 - 5 + 2 * 0) / 1 + 1 = 106
        # 4. pool2の出力サイズ: (106 - 2 + 2 * 0) / 2 + 1 = 53
        output_size = 53

        # input: 40チャンネル * 53(height) * 53(width)
        # output: 2クラスに分類
        self.fc1 = nn.Linear(40 * output_size * output_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
