from tensorlayerx.nn import Module
from tensorlayerx.nn import Conv2d, BatchNorm2d
from tensorlayerx import ReLU

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