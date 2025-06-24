from tensorlayerx.nn import Module
from tensorlayerx.nn import Conv2d, BatchNorm2d, SubpixelConv2d, Flatten, Sequential, BatchNorm
from tensorlayerx.nn import Linear
from tensorlayerx import LeakyReLU, ReLU

W_init = tlx.initializers.TruncatedNormal(stddev=0.02)
G_init = tlx.initializers.TruncatedNormal(mean=1.0, stddev=0.02)

class Discriminator(Module):
    def __init__(self, default_multiplier = 128):
        super(Discriminator, self).__init__()
        self.conv1 = Conv2d(
            out_channels = default_multiplier * 1, kernel_size = (3, 3),
            act=LeakyReLU, W_init=W_init,
            data_format='channels_first', name = "D_conv2d_1_1"
        )

        self.conv2 = Conv2d(
            out_channels = default_multiplier * 1, kernel_size=(3, 3),
            act=LeakyReLU, W_init=W_init,
            data_format='channels_first', b_init=None, name = "D_conv2d_1_2"
        )

        self.bn1 = BatchNorm(
            gamma_init = G_init, data_format = "channels_first",
            name = "D_bn_1"
        )

        self.conv3 = Conv2d(
            out_channels = default_multiplier * 2, kernel_size=(3, 3), stride = (2, 2),
            act=LeakyReLU, W_init=W_init,
            data_format='channels_first', b_init=None, name = "D_conv2d_2_1"
        )

        self.bn2 = BatchNorm(
            gamma_init = G_init, data_format = "channels_first",
            name = "D_bn_2"
        )

        self.conv4 = Conv2d(
            out_channels = default_multiplier * 4, kernel_size=(5, 5), stride = (2, 2),
            act=LeakyReLU, W_init=W_init,
            data_format='channels_first', b_init=None, name = "D_conv2d_2_2"
        )

        self.bn3 = BatchNorm(
            gamma_init = G_init, data_format = "channels_first",
            name = "D_bn_3"
        )

        self.conv5 = Conv2d(
            out_channels = default_multiplier * 8, kernel_size=(7, 7), stride = (2, 2), act=LeakyReLU, W_init=W_init,
            data_format='channels_first', b_init=None, name = "D_conv2d_3_1"
        )

        self.bn4 = BatchNorm(
            gamma_init = G_init, data_format = "channels_first",
            name = "D_bn_4"
        )

        self.conv6 = Conv2d(
            out_channels = default_multiplier * 16, kernel_size=(1, 1), act=LeakyReLU, W_init=W_init,
            data_format='channels_first', b_init=None, name = "D_conv2d_3_2"
        )

        self.bn5 = BatchNorm(
            gamma_init = G_init, data_format = "channels_first",
            name = "D_bn_5"
        )

        self.conv7 = Conv2d(
            out_channels = default_multiplier * 8, kernel_size=(3, 3), stride = (2, 2), act=LeakyReLU, W_init=W_init,
            data_format='channels_first', b_init=None, name = "D_conv2d_4_1"
        )

        self.bn6 = BatchNorm(
            gamma_init = G_init, data_format = "channels_first",
            name = "D_bn_6"
        )

        self.conv8 = Conv2d(
            out_channels = default_multiplier * 1, kernel_size=(1, 1), stride = (1, 1), act=LeakyReLU, W_init=W_init,
            data_format='channels_first', b_init=None, name = "D_conv2d_4_2"
        )

        self.bn7 = BatchNorm(
            gamma_init = G_init, data_format = "channels_first",
            name = "D_bn_7"
        )

        self.flat = Flatten(name = "flat")

        self.dense = Linear(out_features=1, W_init=W_init, name = "output_D", act = tlx.Sigmoid)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = self.bn3(x)

        x = self.conv5(x)
        x = self.bn4(x)

        x = self.conv6(x)
        x = self.bn5(x)

        x = self.conv7(x)
        x = self.bn6(x)

        x = self.conv8(x)
        x = self.bn7(x)

        x = self.flat(x)
        x = self.dense(x)

        return x