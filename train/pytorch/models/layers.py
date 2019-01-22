import math
from decimal import *
import torch
from torch.autograd import Function
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.distributions import Bernoulli


class SampleFn(Function):
    @staticmethod
    def forward(ctx, input):
        # Binary sampling: generate a -1/1 mask based on the latent distribution
        # defined by input
        m = Bernoulli(torch.sigmoid(input))
        return (m.sample() - 0.5) * 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class RoundFn(Function):
    @staticmethod
    def forward(ctx, input, pwr_coef):
        #return (input * (pwr_coef - 1)).round() / (pwr_coef - 1)
        return (input * (pwr_coef - 0.5)).round() / (pwr_coef - 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class QuantizeFn(Function):
    @staticmethod
    def forward(ctx, input, pwr_coef):
        return (input / 2 ** pwr_coef).floor() * 2 ** pwr_coef

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class BinarizeFn(Function):
    @staticmethod
    def forward(ctx, input):
        return ((input > 0).float() - 0.5) * 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class TernarizeFn(Function):
    @staticmethod
    def forward(ctx, input):
        # mag = max(np.abs(input.max()), np.abs(input.min()))
        # return ((input > 0.3*mag).float() - (input < -0.3*mag).float())
        # scale param range, moving average of prev max
        return ((input > 0.7).float() - (input < -0.7).float())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Threebits(Function):
    @staticmethod
    def forward(ctx, input):
        mean = input.abs().mean()

        return ((input > (3. * mean / 4.)).float() * 2 + (input > (1.5 * mean / 4.)).float() * 1 +
                (input > (0.7 * mean / 4.)).float()) - ((input < (-3. * mean / 4.)).float() * 2 +
                                                    (input < (-1.5 * mean / 4.)).float() * 1 + (
                                                               input < (-0.7 * mean / 4.)).float())
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HigherBitsFn(Function):
    @staticmethod
    def forward(ctx, input, bits):
        # Takes input, scales it from -1 to 1 and quantizes it to 2 ** bits - 1 steps

        maxVal = input.abs().max()
        divCuts = ((2 ** bits - 2) / 2)
        return (input / maxVal * divCuts).round() / divCuts

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class MaskConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, mask_bit=1):
        super(MaskConv2d, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = (kernel_size, kernel_size)

        self.mask_val = Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.mask_val.data.normal_()

        self.coef = Parameter(
            torch.Tensor(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.mask_bit = mask_bit

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.coef.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # generate mask
        if self.mask_bit == 1:
            self.mask = BinarizeFn.apply(self.mask_val)
        elif self.mask_bit == 2:
            self.mask = TernarizeFn.apply(self.mask_val)
        elif self.mask_bit == 3:
            self.mask = Threebits.apply(self.mask_val)
        else:
            self.mask = HigherBitsFn.apply(self.mask_val, self.mask_bit)
        self.weight = self.mask * self.coef
        return F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups)


class ActQuant(nn.Module):
    def __init__(self, act_bit=5, scale_coef=10.):
        super(ActQuant, self).__init__()
        self.pwr_coef = 2**act_bit
        self.scale_coef = Parameter(torch.ones(1) * scale_coef)

    def forward(self, x):
        out = F.relu(x)
        out = 0.5 * (out.abs() - (out - self.scale_coef).abs() + self.scale_coef)
        out = RoundFn.apply(out / self.scale_coef, self.pwr_coef) * self.scale_coef
        #out = RoundFn.apply(out / self.scale_coef, self.pwr_coef) * self.pwr_coef //new version
        return out


class ConvBnBias(nn.Module):
    def __init__(
        self, in_planes, planes, kernel_size=3, stride=1, padding=1,
        conv=MaskConv2d, merge=False):
        super(ConvBnBias, self).__init__()
        self.merge = merge
        if not merge:
            self.conv = conv(
                in_planes, planes, kernel_size=kernel_size, stride=stride,
                padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(planes, affine=False)
            self.bias = Parameter(torch.zeros((1, planes, 1, 1)))
        else:
            self.conv = conv(
                in_planes, planes, kernel_size=kernel_size, stride=stride,
                padding=padding, bias=True)

    def forward(self, x):
        if not self.merge:
            return self.bn(self.conv(x)) + self.bias
        else:
            return self.conv(x)


class ConvBnBias2(nn.Module):
    def __init__(
            self, in_planes, planes, kernel_size=3, stride=1, padding=1,
            conv=MaskConv2d, merge=False):
        super(ConvBnBias2, self).__init__()
        self.merge = merge
        self.conv = conv(
            in_planes, planes, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=True)
        if not self.merge:
            self.bn = nn.BatchNorm2d(planes, affine=True)

    def forward(self, x):
        if not self.merge:
            return self.bn(self.conv(x))
        else:
            return self.conv(x)

