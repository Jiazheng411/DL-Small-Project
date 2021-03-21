import torch
from torch import nn
from torch.nn import functional as F


class NetInfected(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 24, 3, 1)
        self.conv2 = nn.Conv2d(24, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2*2*128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, 1, 150, 150)
        x = self.bn(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x) + identity)
        return x


class NetCovid(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.conv1a = nn.Conv2d(1, 24, 3, padding=1)
        self.conv1b = nn.Conv2d(24, 64, 3, padding=1)
        self.conv2 = ResidualBlock(64)
        self.conv3 = ResidualBlock(64)
        self.conv4 = ResidualBlock(64)
        self.conv5 = ResidualBlock(64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64*3*3, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        x = x.view(-1, 1, 150, 150)  # 1x150x150
        x = self.bn(x)
        x = F.relu(self.conv1a(x))   # 24x150x150
        x = F.relu(self.conv1b(x))   # 64x150x150
        x = F.max_pool2d(x, (3, 3))  # 64x50x50
        x = F.max_pool2d(self.conv2(x), (2, 2))  # 64x25x25
        x = F.max_pool2d(self.conv3(x), (2, 2))  # 64x12x12
        x = F.max_pool2d(self.conv4(x), (2, 2))  # 64x6x6
        x = F.max_pool2d(self.conv5(x), (2, 2))  # 64x3x3
        x = x.view(-1, 64*3*3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class NetCovidVisual(nn.Module):
    """ Same as NetCovid, but with feature map recorded """

    def __init__(self, original_model):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.conv1a = nn.Conv2d(1, 24, 3, padding=1)
        self.conv1b = nn.Conv2d(24, 64, 3, padding=1)
        self.conv2 = ResidualBlock(64)
        self.conv3 = ResidualBlock(64)
        self.conv4 = ResidualBlock(64)
        self.conv5 = ResidualBlock(64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64*3*3, 120)
        self.fc2 = nn.Linear(120, 1)
        self.feature_maps = [None] * 4
        self.load_state_dict(original_model.state_dict())

    def forward(self, x):
        x = x.view(-1, 1, 150, 150)  # 1x150x150
        x = self.bn(x)
        x = F.relu(self.conv1a(x))   # 24x150x150
        x = F.relu(self.conv1b(x))   # 64x150x150
        self.feature_maps[0] = x
        x = F.max_pool2d(x, (3, 3))  # 64x50x50
        x = F.max_pool2d(self.conv2(x), (2, 2))  # 64x25x25
        self.feature_maps[1] = x
        x = F.max_pool2d(self.conv3(x), (2, 2))  # 64x12x12
        self.feature_maps[2] = x
        x = F.max_pool2d(self.conv4(x), (2, 2))  # 64x6x6
        self.feature_maps[3] = x
        x = F.max_pool2d(self.conv5(x), (2, 2))  # 64x3x3
        x = x.view(-1, 64*3*3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
