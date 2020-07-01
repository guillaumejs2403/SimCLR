import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fcn = nn.Linear(in_features=hidden_size, out_features=1024)
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv1_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.upsampel = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, imgs):
        x = imgs
        x = self.fcn(x)
        x = x.view(x.size(0), 64, 4, 4)

        x = self.upsampel(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.upsampel(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)

        x = self.upsampel(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)

        return x
