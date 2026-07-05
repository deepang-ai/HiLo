import torch


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


from src.model.HiLo.HiLo_Block import HiLoBlock
from src.model.HiLo.TTA_Mechanism import TTA
import torch.nn as nn


class HiLo(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channel = in_channels
        self.n_classes = out_channels
        super(HiLo, self).__init__()

        self.down_hilo_1 = HiLoBlock(
            64,
            num_heads=8,
            attention_head=1
        )
        self.down_hilo_2 = HiLoBlock(
            128,
            num_heads=8,
            attention_head=1
        )
        self.down_hilo_3 = HiLoBlock(
            256,
            num_heads=8,
            attention_head=1
        )
        self.down_hilo_4 = HiLoBlock(
            512,
            num_heads=8,
            attention_head=1
        )

        # self.up_hilo_1 = HiLoBlock(
        #     64,
        #     num_heads=8,
        #     attention_head=1
        # )
        # self.up_hilo_2 = HiLoBlock(
        #     128,
        #     num_heads=8,
        #     attention_head=1
        # )
        # self.up_hilo_3 = HiLoBlock(
        #     256,
        #     num_heads=8,
        #     attention_head=1
        # )
        # self.up_hilo_4 = HiLoBlock(
        #     512,
        #     num_heads=8,
        #     attention_head=1
        # )

        # 所有 encoder 都使用 padding=1 以保持空间尺寸不变（仅由 pooling 改变）
        self.ec0 = self.encoder(self.in_channel, 32, bias=False, batchnorm=False)
        self.ec1 = self.encoder(32, 64, bias=False, batchnorm=False)
        self.ec2 = self.encoder(64, 64, bias=False, batchnorm=False)
        self.ec3 = self.encoder(64, 128, bias=False, batchnorm=False)
        self.ec4 = self.encoder(128, 128, bias=False, batchnorm=False)
        self.ec5 = self.encoder(128, 256, bias=False, batchnorm=False)
        self.ec6 = self.encoder(256, 256, bias=False, batchnorm=False)
        self.ec7 = self.encoder(256, 512, bias=False, batchnorm=False)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.TTA0 = TTA(num_channels=64, k=3, m=3, c=-0.5)
        self.TTA1 = TTA(num_channels=128, k=3, m=3, c=-0.5)
        self.TTA2 = TTA(num_channels=256, k=3, m=3, c=-0.5)
        self.TTA3 = TTA(num_channels=512, k=3, m=3, c=-0.5)

        # Decoder: 上采样和卷积层
        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(64, out_channels, kernel_size=1, stride=1, bias=False)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,  # ← 关键修改：padding=1
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        down0 = self.ec1(e0)
        down0 = self.down_hilo_1(down0)

        syn0 = self.TTA0(down0)

        e1 = self.pool0(down0)
        e2 = self.ec2(e1)
        down1 = self.ec3(e2)
        down1 = self.down_hilo_2(down1)

        syn1 = self.TTA1(down1)
        del e0, e1, e2, down0

        e3 = self.pool1(down1)
        e4 = self.ec4(e3)
        down2 = self.ec5(e4)
        down2 = self.down_hilo_3(down2)

        syn2 = self.TTA2(down2)
        del e3, e4, down1

        e5 = self.pool2(down2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        e7 = self.down_hilo_4(e7)

        e7 = self.TTA3(e7)
        del e5, e6, down2

        # 拼接上采样结果和跳跃连接，显式指定 dim=1
        d9 = torch.cat((self.dc9(e7), syn2), dim=1)  # [B,512,24,24,24] + [B,256,24,24,24] = [B,768,24,24,24]
        del e7, syn2

        d8 = self.dc8(d9)

        # d8 = self.up_hilo_3(d8)

        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)  # [B,256,48,48,48] + [B,128,48,48,48] = [B,384,48,48,48]
        del d7, syn1

        d5 = self.dc5(d6)

        # d5 = self.up_hilo_2(d5)

        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)  # [B,128,96,96,96] + [B,64,96,96,96] = [B,192,96,96,96]
        del d4, syn0

        d2 = self.dc2(d3)

        # d2 = self.up_hilo_1(d2)

        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        del d1
        return d0



if __name__ == "__main__":
    device = 'cuda:1'
    model = HiLo(in_channels=1, out_channels=2).to(device)
    x = torch.randn(size=(2, 1, 96, 96, 96)).to(device)
    print(model(x).shape)
