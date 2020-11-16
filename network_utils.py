# adapted from https://github.com/facebookresearch/pytorch_GAN_zoo

import math
import torch


def mini_batch_std_dev(x, sub_group_size=4):
    size = x.size()
    sub_group_size = min(size[0], sub_group_size)
    if size[0] % sub_group_size != 0:
        sub_group_size = size[0]
    g = int(size[0] / sub_group_size)
    if sub_group_size > 1:
        y = x.view(-1, sub_group_size, size[1], size[2], size[3])
        y = torch.var(y, 1)
        y = torch.sqrt(y + 1e-8)
        y = y.view(g, -1)
        y = torch.mean(y, 1).view(g, 1)
        y = y.expand(g, size[2] * size[3]).view((g, 1, 1, size[2], size[3]))
        y = y.expand(g, sub_group_size, -1, -1, -1)
        y = y.contiguous().view((-1, 1, size[2], size[3]))
    else:
        y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

    return torch.cat([x, y], dim=1)


def isinf(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return tensor.abs() == math.inf


def isnan(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return tensor != tensor


def finite_check(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    for p in parameters:
        inf_grads = isinf(p.grad.data)
        p.grad.data[inf_grads] = 0

        nan_grads = isnan(p.grad.data)
        p.grad.data[nan_grads] = 0


def wgangp_gradient_penalty(batch_x, batch_y, fake, size, discriminator, weight, backward=True):
    batch_size = batch_y.size(0)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(batch_y.nelement() / batch_size)).contiguous().view(batch_y.size())
    alpha = alpha.to(batch_y.device)
    interpolates = alpha * batch_y + ((1 - alpha) * fake)

    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    decision_interpolate = discriminator(batch_x, interpolates, size, False)
    decision_interpolate = decision_interpolate[:, 0].sum()

    gradients = torch.autograd.grad(outputs=decision_interpolate, inputs=interpolates, create_graph=True,
                                    retain_graph=True)

    gradients = gradients[0].view(batch_size, -1)
    gradients = (gradients * gradients).sum(dim=1).sqrt()
    gradient_penalty = ((gradients - 1.0) ** 2).sum() * weight

    if backward:
        gradient_penalty.backward(retain_graph=True)
    else:
        return gradient_penalty

    return gradient_penalty.item()


class WGANGP:
    def __init__(self, device):
        self.device = device
        self.generation_activation = None
        self.size_decision_layer = 1

    @staticmethod
    def get_criterion(x, status):
        if status:
            return -x[:, 0].sum()
        return x[:, 0].sum()
