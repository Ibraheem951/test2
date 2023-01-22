import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.layer2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3)
        self.layer3 = nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=2)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        y = self.layer3(x)
        out = x + y
        return out

model = Block(3, 6)
input = torch.rand(1,3,30,30)
out = model(input)
print(out.shape)