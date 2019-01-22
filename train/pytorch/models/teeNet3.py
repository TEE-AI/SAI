from .layers import MaskConv2d, ActQuant
import torch.nn as nn
from .teeNet1 import VGGBlock, VGGBlock2

class VGGFC(nn.Module):
    def __init__(self, cfg, module_type, num_classes=2, convs=[nn.Conv2d]*6, quants=[ActQuant]*6):
        super(VGGFC, self).__init__()
        self.num_classes = int(num_classes)

        self.layer1 = self.make_layers(VGGBlock, module_type,  cfg[0], convs[0], quants[0])
        self.layer2 = self.make_layers(VGGBlock, module_type,  cfg[1], convs[1], quants[1])
        self.layer3 = self.make_layers(VGGBlock, module_type,  cfg[2], convs[2], quants[2])
        self.layer4 = self.make_layers(VGGBlock, module_type,  cfg[3], convs[3], quants[3])
        self.layer5 = self.make_layers(VGGBlock, module_type,  cfg[4], convs[4], quants[4])
        self.layer6 = self.make_layers2(VGGBlock2, module_type, [256, 256, 256, self.num_classes], convs[5], quants[5])
        self.classifier = nn.Sequential(
            nn.Softmax(dim=1),
        )

    def make_layers(self, block, module_type, cfg, conv, quant):
        layers = []
        in_channels = cfg[0]
        for layer in cfg[1:]:
            layers.append(block([in_channels, layer], module_type, 1, conv, quant))
            in_channels = layer
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def make_layers2(self, block, module_type, cfg, conv, quant):
        layers = []
        in_channels = cfg[0]
        for layer in cfg[1:]:
            layers.append(block([in_channels, layer], module_type, 1, conv, quant))
            in_channels = layer
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
        out = self.layer6(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

cfg = [[3, 64, 64], [64, 128, 128], [128, 256, 256, 256], [256, 256, 256, 256], [256, 256, 256, 256]]

def VGGWrapperFC(cfg, module_type,  num_classes=40, mask_bits=[1]*6, act_bits=[5]*6):
    assert len(mask_bits) == 6, 'needs masks for 6 major layers'
    assert len(act_bits) == 6, 'needs activations for 6 major layers'

    convs = []
    quants = []

    for mask, act in zip(mask_bits, act_bits):
        if mask == 0:
            conv = nn.Conv2d
        else:
            conv = lambda in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, mask_bit=mask: \
                MaskConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, mask_bit)
        convs.append(conv)

        if act == 0:
            quant = nn.Sequential
        else:
            quant = lambda: ActQuant(act_bit=act)
        quants.append(quant)

    return VGGFC(cfg, module_type, num_classes=num_classes, convs=convs, quants=quants)

def TEE_VGG16_NoFC(module_type, num_classes=2, mask_bits = [1]*5, act_bits = [5]*5):
    return VGGWrapperFC(cfg, module_type, num_classes, mask_bits, act_bits)
