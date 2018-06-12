"""CycleGAN, implemented in Gluon"""
from __future__ import division

from mxnet import nd
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock

__all__ = ['Generator']


class ConvolutionBlock(HybridBlock):
    """Basic Convolution Block"""

    def __init__(self, kernal_size, filters,
                 strides=1, norm_layer=nn.BatchNorm(),
                 activation=nn.Activation('relu')):
        super(ConvolutionBlock, self).__init__()
        self.net = nn.HybridSequential()
        if (isinstance(strides, int)):
            self.net.add(
                nn.Conv2D(filters, kernal_size,
                          strides=strides,
                          padding=((kernal_size - 1) >> 1)))
        else:
            strides = int(round(1 / strides))
            self.net.add(
                nn.Conv2DTranspose(filters, kernal_size,
                                   strides=strides,
                                   padding=((kernal_size - 1) >> 1)))
        if norm_layer is not None:
            self.net.add(norm_layer)

        self.net.add(activation)

    def hybrid_forward(self, F, x):
        return self.net(x)


class ResidualBlock(HybridBlock):
    """Residual Block"""

    def __init__(self, kernal_size, filters,
                 strides=1, norm_layer=nn.BatchNorm):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2D(filters, kernal_size, strides=1, padding=1)
        self.bn1 = norm_layer()
        self.relu = nn.Activation('relu')
        self.conv2 = nn.Conv2D(filters, kernal_size, strides=1, padding=1)
        self.bn2 = norm_layer()

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)

        return out


class Sigmoid(HybridBlock):
    """Sigmoid"""

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, F, x):
        return 1 / (1 + nd.exp(-x))


class Generator(HybridBlock):
    """Generator using Cycle-Consistent Adversarial Networks for Unpaired Image-to-Image Translation

    Parameters
    ----------
    blocks : int
        Number of Convolution Blocks in the generator.


    Reference:

        Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ArXiv:1703.10593 [Cs]. Retrieved from http://arxiv.org/abs/1703.10593

    Examples
    --------
    >>> model = Generator(blocks=6)
    >>> print(model)
    """

    def __init__(self, blocks):
        # Origional paper use 6/9 residual blocks for different resolution
        super(Generator, self).__init__()
        # Follow the naming in the Appendix of the paper
        self.net = nn.HybridSequential()
        self.net.add(
            ConvolutionBlock(
                7, 32,
                strides=1, norm_layer=nn.InstanceNorm(),
                activation=nn.Activation('relu')
            ),  # c7s1_32
            ConvolutionBlock(
                3, 64,
                strides=2, norm_layer=nn.InstanceNorm(),
                activation=nn.Activation('relu')
            ),  # d64
            ConvolutionBlock(
                3, 128,
                strides=2, norm_layer=nn.InstanceNorm(),
                activation=nn.Activation('relu')
            ))  # d128
        for _ in range(blocks):
            self.net.add(
                ResidualBlock(3, 128, norm_layer=nn.InstanceNorm))  # d128

        self.net.add(
            ConvolutionBlock(
                3, 64,
                strides=1 / 2, norm_layer=nn.InstanceNorm(),
                activation=nn.Activation('relu')
            ),  # u64
            ConvolutionBlock(
                3, 32,
                strides=1 / 2, norm_layer=nn.InstanceNorm(),
                activation=nn.Activation('relu')
            ),  # u32
            ConvolutionBlock(
                7, 3,
                strides=1, norm_layer=nn.InstanceNorm(),
                activation=nn.Activation('relu')
            ))  # c7s1_3

    def hybrid_forward(self, F, x):
        return self.net(x)


class Discriminator(HybridBlock):
    """Discriminator in Cycle GAN"""

    def __init__(self, use_sigmoid=False):
        super(Discriminator, self).__init__()
        self.net = nn.HybridSequential()
        self.net.add(
            ConvolutionBlock(
                4, 64,
                strides=2, norm_layer=None,
                activation=nn.LeakyReLU(0.2)
            ),  # C64, without InstanceNorm
            ConvolutionBlock(
                4, 128,
                strides=2, norm_layer=nn.InstanceNorm(),
                activation=nn.LeakyReLU(0.2)
            ),  # C128
            ConvolutionBlock(
                4, 256,
                strides=2, norm_layer=nn.InstanceNorm(),
                activation=nn.LeakyReLU(0.2)
            ),  # C256
            ConvolutionBlock(
                4, 512,
                strides=2, norm_layer=nn.InstanceNorm(),
                activation=nn.LeakyReLU(0.2)
            ),  # C512
            nn.Conv2D(1, 4, strides=1, padding=((4 - 1) >> 1))
        )
        if use_sigmoid:
            self.net.add(Sigmoid())

    def hybrid_forward(self, F, x):
        return self.net(x)
