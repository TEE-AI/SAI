from .layers import MaskConv2d, ActQuant, ConvBnBias, ConvBnBias2
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    def __init__(self, cfg, module_type, stride=1, conv=MaskConv2d, quant=ActQuant):
        super(VGGBlock, self).__init__()

        if module_type!=0:
           self.conv1 = ConvBnBias2(cfg[0], cfg[1], kernel_size=3, stride=stride, padding=1, conv=conv)
        else:  # default [conv(w/0 bias) + bn + bias]
           self.conv1 = ConvBnBias(cfg[0], cfg[1], kernel_size=3, stride=stride, padding=1, conv=conv)

        self.quant1 = quant()

    def forward(self, x):
        out = self.quant1(F.relu(self.conv1(x)))
        return out


class VGGBlock2(nn.Module):
    def __init__(self, cfg, module_type, stride=1, conv=MaskConv2d, quant=ActQuant):
        super(VGGBlock2, self).__init__()
    
        if module_type!=0:
           self.conv1 = ConvBnBias2(cfg[0], cfg[1], kernel_size=3, stride=stride, padding=0, conv=conv)
        else:  # default [conv(w/0 bias) + bn + bias]
           self.conv1 = ConvBnBias(cfg[0], cfg[1], kernel_size=3, stride=stride, padding=0, conv=conv)

        self.quant1 = quant()

    def forward(self, x):
        out = self.quant1(F.relu(self.conv1(x)))
        return out


class VGG16(nn.Module):
    def __init__(self, cfg, module_type, num_classes=40, convs=[nn.Conv2d]*5, quants=[ActQuant]*5):
        super(VGG16, self).__init__()
        self.num_classes = int(num_classes)

        self.layer1 = self.make_layers(VGGBlock, module_type, cfg[0], convs[0], quants[0])
        self.layer2 = self.make_layers(VGGBlock, module_type, cfg[1], convs[1], quants[1])
        self.layer3 = self.make_layers(VGGBlock, module_type, cfg[2], convs[2], quants[2])
        self.layer4 = self.make_layers(VGGBlock, module_type, cfg[3], convs[3], quants[3])
        self.layer5 = self.make_layers(VGGBlock, module_type, cfg[4], convs[4], quants[4])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),
        )


    def make_layers(self, block, module_type, cfg, conv, quant):
        layers = []
        in_channels = cfg[0]
        for layer in cfg[1:]:
            layers.append(block([in_channels, layer], module_type, 1, conv, quant))
            in_channels = layer
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x *= 255
        x /= 8
        x -= 0.499
        x = x.round()
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

cfg = [[3, 64, 64], [64, 128, 128], [128, 256, 256, 256], [256, 512, 512, 512], [512, 512, 512, 512]]

def VGGWrapper(cfg, module_type, num_classes=40, mask_bits=[1] * 5, act_bits=[5]*5):
    assert len(mask_bits) == 5, 'needs masks for 5 major layers'
    assert len(act_bits) == 5, 'needs activations for 5 major layers'

    convs = []
    quants = []

    for mask, act in zip(mask_bits, act_bits):
        if mask == 0:
            conv = nn.Conv2d
        else:
            conv = lambda in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, mask_bit=mask: \
                MaskConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                           mask_bit)
        convs.append(conv)
        if act == 0:
            quant = nn.Sequential
        else:
            quant = lambda: ActQuant(act_bit=act)
        quants.append(quant)

    return VGG16(cfg, module_type, num_classes, convs, quants)


def TEE_VGG16(module_type, num_classes=40, mask_bits=[1]*5, act_bits=[5]*5):
    return VGGWrapper(cfg, module_type, num_classes, mask_bits=mask_bits, act_bits=act_bits)

