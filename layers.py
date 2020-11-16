# adapted from https://github.com/facebookresearch/pytorch_GAN_zoo

import math
import torch.nn as nn
from numpy import prod


def flatten(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return x.view(-1, num_features)


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x


class ConstrainedLayer(nn.Module):
    def __init__(self, module, equalized=True, lr_mul=1.0, init_bias_to_zero=True):
        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if init_bias_to_zero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lr_mul
            self.weight = self.get_layer_normalization_factor(self.module) * lr_mul

    @staticmethod
    def get_layer_normalization_factor(x):
        size = x.weight.size()
        fan_in = prod(size[1:])
        return math.sqrt(2.0 / fan_in)

    def forward(self, x):

        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedConv2d(ConstrainedLayer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True, **kwargs):
        ConstrainedLayer.__init__(self, nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias),
                                  **kwargs)


class EqualizedLinear(ConstrainedLayer):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        ConstrainedLayer.__init__(self, nn.Linear(in_features, out_features, bias=bias), **kwargs)


class NormalizationLayer(nn.Module):
    def __init__(self):
        super(NormalizationLayer, self).__init__()

    @staticmethod
    def forward(x, epsilon=1e-8):
        return x * (((x ** 2).mean(dim=1, keepdim=True) + epsilon).rsqrt())
