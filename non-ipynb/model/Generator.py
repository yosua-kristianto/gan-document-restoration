from tensorlayerx.nn import Module
from tensorlayerx.nn import Conv2d, BatchNorm2d, SubpixelConv2d, Flatten, Sequential, BatchNorm
from tensorlayerx.nn import Linear
from tensorlayerx import LeakyReLU, ReLU

class ResidualBlock(Module):

    def __init__(self, resblock_number: int = 1):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2d(
            out_channels=128, kernel_size=(3, 3), stride=(1, 1),
            act=ReLU, padding='SAME',
            data_format='channels_first', b_init=None, name = f"conv2d_resblock_{resblock_number}_1"
        )

        self.bn1 = BatchNorm2d(
            num_features=128, act=None,
            data_format='channels_first', name = f"batchnorm2d_resblock_{resblock_number}_1"
        )

        self.conv2 = Conv2d(
            out_channels=128, kernel_size=(3, 3), stride=(1, 1),
            act=ReLU, padding='SAME',
            data_format='channels_first', b_init=None, name = f"conv2d_resblock_{resblock_number}_2"
        )

        self.bn2 = BatchNorm2d(
            num_features=128, act=None,
            data_format='channels_first', name = f"batchnorm2d_resblock_{resblock_number}_2"
        )

    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.conv2(z)
        z = self.bn2(z)
        x = x + z
        return x
    
class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = Conv2d(
            out_channels=128, kernel_size=(3, 3), stride=(1, 1),
            act=ReLU,
            data_format='channels_first', name = "conv2d_G_1"
        )

        self.residual_block = self.make_layer()

        self.conv2 = Conv2d(
            out_channels=128, kernel_size=(3, 3), stride=(1, 1),
            act=ReLU,
            data_format='channels_first', b_init=None, name = "conv2d_G_2"
        )

        self.bn1 = BatchNorm2d(
            num_features=128,
            data_format='channels_first', name = "batchnorm2d_G_1"
        )

        self.conv3 = Conv2d(
            out_channels=512, kernel_size=(3, 3), stride=(1, 1),
            data_format='channels_first', name = "conv2d_G_3"
        )

        self.subpixelconv1 = SubpixelConv2d(
            scale=2, act=ReLU,
            data_format='channels_first', name = "subpixelconv2d_G_1"
        )

        self.conv4 = Conv2d(
            out_channels=512, kernel_size=(3, 3), stride=(1, 1),
            data_format='channels_first', name = "conv2d_G_4"
        )

        self.subpixelconv2 = SubpixelConv2d(
            scale=2, act=ReLU,
            data_format='channels_first', name = "subpixelconv2d_G_2"
        )

        self.output = Conv2d(
            out_channels = 3, kernel_size=(1, 1), stride=(1, 1),
            act=tlx.Tanh,
            data_format='channels_first', name = "conv2d_G_output"
        )

    def make_layer(self):
        layer_list = []

        for i in range(16):
            layer_list.append(ResidualBlock(i))

        return Sequential(layer_list)

    def forward(self, x):
        x = self.conv1(x)
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = x + temp
        x = self.conv3(x)
        x = self.subpixelconv1(x)
        x = self.conv4(x)
        x = self.subpixelconv2(x)
        x = self.output(x)

        return x