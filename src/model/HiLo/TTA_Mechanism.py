import torch


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

import torch.nn as nn

import math

import torch.nn.functional as F


def custom_function(x, k, m, c):
    term_1 = torch.tanh(k * x) + 1
    term_1 = term_1 / 2
    term_2 = 1 / (1 + torch.exp(-m * (x - c)))
    return term_1 * term_2


class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU(inplace=True)
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        # output_tensor = input_tensor + final_squeeze_tensor
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor


class ArctanScaledActivation(nn.Module):
    def __init__(self):
        super(ArctanScaledActivation, self).__init__()

    def forward(self, x):
        return (torch.atan(x) + torch.pi / 2) / torch.pi

# Tri-Axial Telescopic Activation
class TTA(nn.Module):
    def __init__(self, num_channels, k, m, c):
        super(TTA, self).__init__()
        num_channels_reduced = num_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv_l = nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1)  # 可能要改
        self.conv_2 = nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1)
        # self.act = ComplexTanhTransform(k=2)
        # self.act =nn.Tanh()
        # self.act =nn.Softsign()
        # self.act = ArctanScaledActivation()
        self.pooling = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.pae = ProjectExciteLayer(num_channels)

        self.k = k
        self.m = m
        self.c = c

    def forward(self, input_tensor):
        # input_tensor2 = self.pooling(input_tensor2)

        rl = self.pae(input_tensor)
        # final_squeeze_tensor = custom_function(self.conv_l(rl), k=1, m=1, c=-1 * (math.e ** 6))
        final_squeeze_tensor = custom_function(self.conv_l(rl), k=self.k, m=self.m, c=self.c)

        output_tensor = self.conv_2(final_squeeze_tensor)
        output_tensor.add_(input_tensor)
        return output_tensor


if __name__ == "__main__":
    device = 'cuda:5'
    model1 = TTA(num_channels=32).to(device)
    model2 = ProjectExciteLayer(num_channels=32).to(device)

    x = torch.randn(size=(2, 32, 128, 128, 128)).to(device)
    # y = torch.randn(size=(2, 16, 96, 96, 96)).to(device)
    # z= torch.randn(size=(2, 32, 48, 48, 48)).to(device)
    print(model1(x).shape)
    # print(model2(y).shape)
    # print(model1(z,y).shape)
# Number of network parameters: 4118849 Baseline
