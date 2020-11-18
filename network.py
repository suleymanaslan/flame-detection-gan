# adapted from https://github.com/facebookresearch/pytorch_GAN_zoo

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import flatten, upscale2d, EqualizedLinear, EqualizedConv2d, NormalizationLayer
from network_utils import mini_batch_std_dev


class GeneratorFormat(nn.Module):
    def __init__(self, frames_per_clip):
        super(GeneratorFormat, self).__init__()
        self.frames_per_clip = frames_per_clip

    def forward(self, x, size):
        return x.view(x.shape[0], 3, self.frames_per_clip, size, size)


class Generator(nn.Module):
    def __init__(self, frames_per_clip):
        super(Generator, self).__init__()
        self.generator_format = GeneratorFormat(frames_per_clip)

        self.dim_latent = 1024
        self.depth_scale0 = 256
        self.dim_output = 3 * frames_per_clip
        self.equalized_lr = True
        self.init_bias_to_zero = True
        self.scales_depth = [self.depth_scale0]

        self.scale_layers = nn.ModuleList()

        self.to_rgb_layers = nn.ModuleList()
        self.to_rgb_layers.append(EqualizedConv2d(self.depth_scale0, self.dim_output, 1, equalized=self.equalized_lr,
                                                  init_bias_to_zero=self.init_bias_to_zero))

        self.format_layer = EqualizedLinear(self.dim_latent, 16 * self.scales_depth[0], equalized=self.equalized_lr,
                                            init_bias_to_zero=self.init_bias_to_zero)

        self.group_scale0 = nn.ModuleList()
        self.group_scale0.append(
            EqualizedConv2d(self.depth_scale0, self.depth_scale0, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

        self.alpha = 0

        self.leaky_relu = torch.nn.LeakyReLU(0.2, inplace=True)

        self.normalization_layer = NormalizationLayer()

        self.generation_activation = None

    def add_scale(self, depth_new_scale=None):
        if depth_new_scale == None:
            depth_new_scale = self.depth_scale0
        depth_last_scale = self.scales_depth[-1]
        self.scales_depth.append(depth_new_scale)

        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_last_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

        self.to_rgb_layers.append(EqualizedConv2d(depth_new_scale, self.dim_output, 1, equalized=self.equalized_lr,
                                                  init_bias_to_zero=self.init_bias_to_zero))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x, size):
        x = self.normalization_layer(x)
        x = flatten(x)
        x = self.leaky_relu(self.format_layer(x))
        x = x.view(x.size()[0], -1, 4, 4)
        x = self.normalization_layer(x)

        for conv_layer in self.group_scale0:
            x = self.leaky_relu(conv_layer(x))
            x = self.normalization_layer(x)

        if self.alpha > 0 and len(self.scale_layers) == 1:
            y = self.to_rgb_layers[-2](x)
            y = upscale2d(y)

        for scale, layer_group in enumerate(self.scale_layers, 0):
            x = upscale2d(x)
            for conv_layer in layer_group:
                x = self.leaky_relu(conv_layer(x))
                x = self.normalization_layer(x)
            if self.alpha > 0 and scale == (len(self.scale_layers) - 2):
                y = self.to_rgb_layers[-2](x)
                y = upscale2d(y)

        x = self.to_rgb_layers[-1](x)

        if self.alpha > 0:
            x = self.alpha * y + (1.0 - self.alpha) * x

        if self.generation_activation is not None:
            x = self.generation_activation(x)

        x = self.generator_format(x, size)

        return x


class DiscriminatorFormat(nn.Module):
    def __init__(self, frames_per_clip):
        super(DiscriminatorFormat, self).__init__()
        self.frames_per_clip = frames_per_clip
        self.dim_out = 3 * self.frames_per_clip

    def forward(self, x, size):
        return x.view(x.shape[0], 3 * self.frames_per_clip, size, size)


class Discriminator(nn.Module):
    def __init__(self, frames_per_clip):
        super(Discriminator, self).__init__()
        self.discriminator_format = DiscriminatorFormat(frames_per_clip)
        self.dim_input = self.discriminator_format.dim_out

        self.depth_scale0 = 256
        self.size_decision_layer = 1
        self.size_classification_layer = 1
        self.equalized_lr = True
        self.init_bias_to_zero = True
        self.mini_batch_normalization = True
        self.dim_entry_scale0 = self.depth_scale0 + 1
        self.scales_depth = [self.depth_scale0]

        self.scale_layers = nn.ModuleList()

        self.from_rgb_layers = nn.ModuleList()
        self.from_rgb_layers.append(EqualizedConv2d(self.dim_input, self.depth_scale0, 1, equalized=self.equalized_lr,
                                                    init_bias_to_zero=self.init_bias_to_zero))

        self.merge_layers = nn.ModuleList()

        self.decision_layer = EqualizedLinear(self.scales_depth[0], self.size_decision_layer,
                                              equalized=self.equalized_lr, init_bias_to_zero=self.init_bias_to_zero)
        self.classification_layer = nn.Linear(self.scales_depth[0], self.size_classification_layer)

        self.group_scale0 = nn.ModuleList()
        self.group_scale0.append(
            EqualizedConv2d(self.dim_entry_scale0, self.depth_scale0, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))
        self.group_scale0.append(
            EqualizedLinear(self.depth_scale0 * 16, self.depth_scale0, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

        self.alpha = 0

        self.leaky_relu = torch.nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

    def add_scale(self, depth_new_scale=None):
        if depth_new_scale == None:
            depth_new_scale = self.depth_scale0
        depth_last_scale = self.scales_depth[-1]
        self.scales_depth.append(depth_new_scale)

        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_last_scale, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

        self.from_rgb_layers.append(
            EqualizedConv2d(self.dim_input, depth_new_scale, 1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x, size, get_classification=False, get_feature=False):
        x = self.discriminator_format(x, size)

        if self.alpha > 0 and len(self.from_rgb_layers) > 1:
            y = F.avg_pool2d(x, (2, 2))
            y = self.leaky_relu(self.from_rgb_layers[-2](y))

        x = self.leaky_relu(self.from_rgb_layers[-1](x))

        merge_layer = self.alpha > 0 and len(self.scale_layers) > 1

        shift = len(self.from_rgb_layers) - 2

        for group_layer in reversed(self.scale_layers):
            for layer in group_layer:
                x = self.leaky_relu(layer(x))

            x = nn.AvgPool2d((2, 2))(x)

            if merge_layer:
                merge_layer = False
                x = self.alpha * y + (1 - self.alpha) * x

            shift -= 1

        if self.mini_batch_normalization:
            x = mini_batch_std_dev(x)

        x = self.leaky_relu(self.group_scale0[0](x))

        x = flatten(x)

        x = self.leaky_relu(self.group_scale0[1](x))

        if get_classification:
            out = self.sigmoid(self.classification_layer(x))
        else:
            out = self.decision_layer(x)

        if not get_feature:
            return out

        return out, x
