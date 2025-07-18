import torch
import torch.nn.functional as F
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class nnUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # Down sampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        prev_channels = features[-1]
        self.bottleneck = DoubleConv(prev_channels, prev_channels * 2)

        skip_ch_list = features[::-1]
        up_in_ch_list = [features[-1]*2] + skip_ch_list[:-1]
        up_channels = list(zip(up_in_ch_list, skip_ch_list))

        for up_in_ch, skip_ch in up_channels:
            self.ups.append(nn.ConvTranspose2d(up_in_ch, skip_ch, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(skip_ch * 2, skip_ch))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.ups)):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx]
            if idx == 0 and self.training and not hasattr(self, "_debug_printed"):
                print(f"[DEBUG] upsampled x: {x.shape}, skip: {skip_connection.shape}")
                x_tmp = torch.cat((skip_connection, x), dim=1)
                print(f"[DEBUG] concat x: {x_tmp.shape}")
                self._debug_printed = True
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.up_convs[idx](x)

        return torch.sigmoid(self.final_conv(x))