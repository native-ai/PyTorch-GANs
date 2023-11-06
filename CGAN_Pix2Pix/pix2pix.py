import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = stride, bias = False, padding_mode = "reflect"),
            nn.BatchNorm2d(out_channels), # TODO: will be changed to nn.InstanceNorm2d()
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv_block(x)


class Discriminator(nn.Module):
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

# def disc_test():
#     x = torch.randn([1, 3, 256, 256])
#     y = torch.randn([1, 3, 256, 256])
#     model = Discriminator(in_channels = 3)
#     preds = model(x, y)
#     print(preds.shape)
#
# if __name__ == "__main__":
#     disc_test()


class CNNBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, activation="relu", use_dropout=False):
        super(CNNBlock2, self).__init__()

        layers = []

        if downsample:
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
            )
        else:
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            )

        if activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.2))
        elif activation == "relu":
            layers.append(nn.ReLU())
        else:
            raise ValueError("Activation should be 'leaky_relu' or 'relu'.")

        if use_dropout:
            layers.append(nn.Dropout(0.5))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, in_channels, features_g = 64):
        super().__init__()
        self.init_down = nn.Sequential(
            nn.Conv2d(in_channels, features_g, kernel_size = 4, stride = 2, padding = 1, padding_mode = "reflect"),
            nn.LeakyReLU(0.2)
        )

        self.first_down = CNNBlock2(features_g, features_g * 2, downsample = True, activation = "leaky_relu", use_dropout = False)
        self.second_down = CNNBlock2(features_g * 2, features_g * 4, downsample = True, activation = "leaky_relu", use_dropout = False)
        self.third_down = CNNBlock2(features_g * 4, features_g * 8, downsample = True, activation = "leaky_relu", use_dropout = False)
        self.fourth_down = CNNBlock2(features_g * 8, features_g * 8, downsample = True, activation = "leaky_relu", use_dropout = False)
        self.fifth_down = CNNBlock2(features_g * 8, features_g * 8, downsample = True, activation = "leaky_relu", use_dropout = False)
        self.sixth_down = CNNBlock2(features_g * 8, features_g * 8, downsample = True, activation = "leaky_relu", use_dropout = False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features_g * 8, features_g * 8, kernel_size = 4, stride = 2, padding = 1, padding_mode = "reflect"),
            nn.ReLU()
        )

        self.first_up = CNNBlock2(features_g * 8, features_g * 8, downsample = False, activation = "relu", use_dropout = True)
        self.second_up = CNNBlock2(features_g * 8 * 2, features_g * 8, downsample = False, activation = "relu", use_dropout = True)
        self.third_up = CNNBlock2(features_g * 8 * 2, features_g * 8, downsample = False, activation = "relu", use_dropout = True)
        self.fourth_up = CNNBlock2(features_g * 8 * 2, features_g * 8, downsample = False, activation = "relu", use_dropout = True)
        self.fifth_up = CNNBlock2(features_g * 8 * 2, features_g * 4, downsample = False, activation = "relu", use_dropout = True)
        self.sixth_up = CNNBlock2(features_g * 4 * 2, features_g * 2, downsample = False, activation = "relu", use_dropout = True)
        self.fifth_up = CNNBlock2(features_g * 2 * 2, features_g, downsample = False, activation = "relu", use_dropout = True)

        self.last_up = nn.Sequential(
            nn.ConvTranspose2d(features_g * 2, in_channels, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh(), # [-1, 1]
        )

    def forward(self, x):
        pass