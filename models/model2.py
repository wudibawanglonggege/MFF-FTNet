import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=320, out_channels=160, kernel_size=2, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.liner = nn.AdaptiveMaxPool1d(201)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pool(F.silu(self.conv1(x)))
        x = F.silu(self.conv2(x))
        x = x.view(x.size(0), x.size(1), -1)
        return self.liner(x).transpose(1, 2)
