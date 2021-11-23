import torch
from torch import nn


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        module = SiLU()
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Focus(nn.Module):
    # 对于原始数据的处理,每四个相邻像素点取一个固定位置，取完后将所取堆叠。得到的效果是长宽变为1/2，通道变为四倍。
    def __init__(self, in_channel, out_channels, ksize=1, stride=1, act='silu'):
        super().__init__()
        self.conv = BaseConv(in_channel * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1)
        return self.conv(x)


class BaseConv(nn.Module):
    # 初始卷积，最底层的卷积层定义
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act='silu'):
        super().__init__()
        pad = (ksize - 1) // 2  # 设置边界补全
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    # Depthwise Convolution的一个卷积核负责一个通道，一个通道只被一个卷积核卷积的特殊卷积形式
    # Pointwise Convolution的卷积核的尺寸为 1×1×通道数
    # Depthwise Separable Convolution为epthwise(DW)卷积与Pointwise(PW)卷积的合并
    def __init__(self, in_channels, out_channels, ksize, stride=1, act='silu'):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))


class SPPBottleneck(nn.Module):
    # spp模块，它的特点是通过不同池化核大小的最大池化进行特征提取,后面进行堆叠，能够提高网络的感受野
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)  # 计算中间层通道数，每经过一次池化会堆叠一个通道的层数
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Bottleneck(nn.Module):
    # 基础的残差网络模块，将经过1*1+3*3卷积，同时有残差结构的直接结合。
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    # csp结构，定义在小的残差网络基础之上。
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # 定义中间隐层层数
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)  # 主干边
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)  # 残差边
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)  # 最后两遍合并
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in
                       range(n)]  # 主干边结构
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act='silu'):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise,
                     act=act),
        )
    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

if __name__ == '__main__':
    print(CSPDarknet(1, 1)(torch.rand(1,3,640,64)).keys())