import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = stride, bias = False, padding_mode = "reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )


    def forward(self, x):
        return self.conv_block(x)


class Discriminator():
    def __init__(self, in_channels = 3, features_d = [64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features_d[0], kernel_size = 4, stride = 2, padding = 1, padding_mode = "reflect"),
            nn.LeakyReLU(0.2)
        )

        disc_layers = []
        in_channels = features_d[0]
        for feature in features_d[1:]:
            disc_layers.append(CNNBlock(in_channels, feature, stride = 1 if feature == features_d[-1] else 2))
            in_channels = feature

        disc_layers.append(nn.Conv2d(in_channels, 1, kernel_size = 4, stride = 1, padding = 1, padding_mode = "reflect"))
        self.model = nn.Sequential(*disc_layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim = 1)
        x = self.initial(x)
        return self.model(x)


def disc_test():
    x = torch.randn([1, 3, 256, 256])
    y = torch.randn([1, 3, 256, 256])
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)

if __name__ == "__main__":
    disc_test()