import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.fft as fft
import torch.nn.functional as F


class HighMixer3D(nn.Module):
    """3D高频率特征混合模块：通过卷积和池化分支提取高频率特征"""

    def __init__(self, in_dim, out_dim=None, kernel_size=3, stride=1, padding=1, **kwargs):
        super().__init__()
        out_dim = out_dim or in_dim  # 默认为输入维度

        # 分割输入通道为卷积分支和池化分支
        self.cnn_in = cnn_in = in_dim // 2
        self.pool_in = pool_in = in_dim - cnn_in  # 确保总和为in_dim

        # 分支输出维度（各翻倍）
        self.cnn_dim = cnn_dim = out_dim // 2
        self.pool_dim = pool_dim = out_dim - cnn_dim

        # 卷积分支：1x1卷积升维 + BatchNorm + 深度可分离卷积 + BatchNorm + GELU
        self.conv1 = nn.Conv3d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(cnn_dim)  # 添加BatchNorm
        self.proj1 = nn.Conv3d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False, groups=cnn_dim)  # 深度可分离卷积
        self.bn2 = nn.BatchNorm3d(cnn_dim)  # 添加BatchNorm
        self.mid_gelu1 = nn.GELU()

        # 池化分支：最大池化 + 1x1卷积升维 + BatchNorm + GELU
        self.Maxpool = nn.MaxPool3d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv3d(pool_in, pool_dim, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(pool_dim)  # 添加BatchNorm
        self.mid_gelu2 = nn.GELU()

    def forward(self, x):
        # 输入形状: (B, C, H, W, D)

        # 卷积分支
        cx = x[:, :self.cnn_in, :, :, :].contiguous()  # 取前半通道
        cx = self.conv1(cx)  # 升维到cnn_dim
        cx = self.bn1(cx)  # BatchNorm
        cx = self.proj1(cx)  # 深度可分离卷积
        cx = self.bn2(cx)  # BatchNorm
        cx = self.mid_gelu1(cx)

        # 池化分支
        px = x[:, self.cnn_in:, :, :, :].contiguous()  # 取后半通道
        px = self.Maxpool(px)  # 池化
        px = self.proj2(px)  # 升维到pool_dim
        px = self.bn3(px)  # BatchNorm
        px = self.mid_gelu2(px)

        # # 拼接并投影到目标维度
        # hx = torch.cat((cx, px), dim=1)  # 通道维度拼接 (B, cnn_dim+pool_dim, H, W, D)

        return cx, px


class LowMixer3D(nn.Module):
    """3D低频率特征混合模块：使用Mamba替代Transformer"""

    def __init__(self, in_dim, out_dim=None, pool_size=4, d_model=None, expand=2, **kwargs):
        super().__init__()
        out_dim = out_dim or in_dim
        self.pool_size = pool_size
        self.d_model = d_model if d_model else in_dim

        # 特征压缩与恢复
        self.pool = nn.AvgPool3d(pool_size, stride=pool_size, padding=0) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size, mode='trilinear',
                                  align_corners=False) if pool_size > 1 else nn.Identity()

        # Mamba块
        self.proj_in = nn.Linear(in_dim, self.d_model) if in_dim != self.d_model else nn.Identity()
        self.bn_in = nn.BatchNorm1d(self.d_model)  # 添加BatchNorm（序列维度）
        self.mamba = Mamba(
            d_model=self.d_model,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj_out = nn.Linear(self.d_model, out_dim) if self.d_model != out_dim else nn.Identity()
        self.bn_out = nn.BatchNorm1d(out_dim)  # 添加BatchNorm（序列维度）

        # 输出投影（3D卷积）
        self.out_proj = nn.Conv3d(out_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        if in_dim != out_dim:
            self.bn_proj = nn.BatchNorm3d(out_dim)  # 添加BatchNorm（3D特征）

    def forward(self, x):
        # 输入形状: (B, C, H, W, D)
        B, C, H, W, D = x.shape

        # 下采样
        xa = self.pool(x)  # (B, C, H/pool_size, W/pool_size, D/pool_size)

        # 重塑为序列 (B, N, C)
        xa = xa.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C)  # (B, N, C)

        # 投影到Mamba维度 + BatchNorm
        xa = self.proj_in(xa)

        xa = self.bn_in(xa.transpose(1, 2)).transpose(1, 2)  # BatchNorm1d需要特征维度在第2位

        # 通过Mamba处理
        xa = self.mamba(xa)

        # 投影回原始维度 + BatchNorm
        xa = self.proj_out(xa)
        xa = self.bn_out(xa.transpose(1, 2)).transpose(1, 2)  # 恢复序列维度

        # 重塑回3D特征图
        pool_size = self.pool_size
        h = H // pool_size if pool_size > 1 else H
        w = W // pool_size if pool_size > 1 else W
        d = D // pool_size if pool_size > 1 else D
        xa = xa.view(B, h, w, d, -1).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, h, w, d)

        # 上采样回原始空间维度
        xa = self.uppool(xa)  # (B, C, H, W, D)

        # 输出投影 + BatchNorm
        if hasattr(self, 'out_proj') and not isinstance(self.out_proj, nn.Identity):
            xa = self.out_proj(xa)
            xa = self.bn_proj(xa)

        return xa


class HiLoBlock(nn.Module):
    """3D混合模块：结合高频率卷积特征和低频率注意力特征"""

    def __init__(self, in_dim, out_dim=None, num_heads=8, proj_drop=0.,
                 attention_head=1, pool_size=2, rate=0.5, mask_strength=1.0, **kwargs):
        super().__init__()
        out_dim = out_dim or in_dim  # 默认为输入维度

        self.in_dim = in_dim

        self.num_heads = num_heads
        self.head_dim = head_dim = in_dim // num_heads  # 每个注意力头的维度

        # 分割输入维度为高频率分支和低频率分支
        self.low_dim = low_dim = attention_head * head_dim  # 低频率分支维度
        self.high_dim = high_dim = in_dim - low_dim  # 高频率分支维度

        # 子模块：高频率混合器和低频率混合器
        self.high_mixer = HighMixer3D(high_dim, out_dim=high_dim * 2)
        self.low_mixer = LowMixer3D(low_dim, d_model=low_dim * 2, expand=2,
                                    pool_size=pool_size, **kwargs)

        # 3D融合卷积 + BatchNorm
        concat_dim = high_dim * 4 + low_dim * 2  # 拼接后的维度
        self.concat_dim = concat_dim
        self.conv_fuse = nn.Conv3d(concat_dim, concat_dim, kernel_size=3,
                                   stride=1, padding=1, bias=False, groups=concat_dim)
        self.bn_fuse = nn.BatchNorm3d(concat_dim)  # 添加BatchNorm
        self.relu = nn.ReLU(inplace=True)

        # 输出投影层 + BatchNorm
        self.proj = nn.Conv3d(concat_dim, out_dim, kernel_size=1)
        self.bn_proj = nn.BatchNorm3d(out_dim)  # 添加BatchNorm
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask_strength = nn.Parameter(torch.tensor(mask_strength), requires_grad=True)

        self.rate = nn.Parameter(torch.tensor(rate), requires_grad=True)

        reduction = max(1, concat_dim // 16)  # 避免 reduction 过大导致维度为0
        self.se_fc1 = nn.Linear(concat_dim, reduction, bias=False)
        self.se_relu = nn.ReLU()
        self.se_fc2 = nn.Linear(reduction, concat_dim, bias=False)
        self.se_sigmoid = nn.Sigmoid()

    def create_3d_lowpass_mask(self, x, cutoff=0.5):
        """创建3D低通掩码（匹配rfftn半谱布局）"""
        B, C, H, W, D = x.shape
        kx = torch.fft.fftfreq(H, d=1.0, device=x.device) * 2.0
        ky = torch.fft.fftfreq(W, d=1.0, device=x.device) * 2.0
        kz = torch.fft.rfftfreq(D, d=1.0, device=x.device) * 2.0

        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        distance = torch.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

        mask = torch.exp(-(distance ** 2) / (2 * cutoff ** 2))
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask * self.mask_strength

    def create_3d_highpass_mask(self, x, cutoff=0.5):
        """创建3D高通掩码（保留高频分量）"""
        lowpass = self.create_3d_lowpass_mask(x, cutoff)
        # 高通掩码 = 1 - 低通掩码
        return (1 - lowpass) * self.mask_strength

    def create_3d_frequency_masks(self, x, cutoff=0.5):
        lowpass = self.create_3d_lowpass_mask(x, cutoff)
        highpass = (1 - lowpass) * self.mask_strength
        return lowpass, highpass

    def apply_frequency_mask(self, x, mask):
        """对特征图应用频域掩码"""
        input_dtype = x.dtype
        work_dtype = torch.float32
        spatial_shape = x.shape[2:]

        x_fft = fft.rfftn(x.to(work_dtype), dim=(2, 3, 4))
        del x

        x_fft.mul_(mask.to(device=x_fft.device, dtype=x_fft.real.dtype))
        del mask

        x_ifft = fft.irfftn(x_fft, s=spatial_shape, dim=(2, 3, 4))
        del x_fft

        out = x_ifft.to(input_dtype)
        del x_ifft
        return out

    def channel_attention(self, x):
        """
        实现3D SE通道注意力。
        输入: (B, C, H, W, D)
        输出: (B, C, 1, 1, 1) 的通道权重
        """
        B, C, H, W, D = x.shape
        # 全局平均池化 -> (B, C, 1, 1, 1)
        y = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(B, C)
        # MLP
        y = self.se_fc1(y)
        y = self.se_relu(y)
        y = self.se_fc2(y)
        y = self.se_sigmoid(y)
        # 扩展回原空间维度
        return y.view(B, C, 1, 1, 1)

    def forward(self, x, visual=False):
        # 输入形状: (B, C, H, W, D)
        B, C, H, W, D = x.shape

        # 生成频域掩码
        lowpass_mask, highpass_mask = self.create_3d_frequency_masks(x)

        x_cat = torch.empty(
            (B, self.concat_dim, H, W, D),
            dtype=x.dtype,
            device=x.device,
        )

        offset = 0

        def _append_feature(tensor):
            nonlocal offset
            channels = tensor.shape[1]
            x_cat[:, offset:offset + channels].copy_(tensor)
            offset += channels

        hx = x[:, :self.high_dim, :, :, :].contiguous()  # 高频率特征
        cx_spc, px_spc = self.high_mixer(hx)
        del hx

        features = None
        if visual:
            cx_fft = self.apply_frequency_mask(cx_spc, highpass_mask)
            px_fft = self.apply_frequency_mask(px_spc, highpass_mask)

            lx = x[:, self.high_dim:, :, :, :].contiguous()  # 低频率特征
            del x
            lx_spc = self.low_mixer(lx)
            del lx
            lx_fft = self.apply_frequency_mask(lx_spc, lowpass_mask)
            del lowpass_mask, highpass_mask

            features = {
                'cx_fea': cx_spc,
                'px_fea': px_spc,
                'cx_fft': cx_fft,
                'px_fft': px_fft,
                'lx_spc': lx_spc,
                'lx_fft': lx_fft,
            }

            _append_feature(cx_spc)
            _append_feature(cx_fft)
            _append_feature(px_spc)
            _append_feature(px_fft)
            _append_feature(lx_spc)
            _append_feature(lx_fft)
        else:
            _append_feature(cx_spc)
            cx_fft = self.apply_frequency_mask(cx_spc, highpass_mask)
            _append_feature(cx_fft)
            del cx_spc, cx_fft

            _append_feature(px_spc)
            px_fft = self.apply_frequency_mask(px_spc, highpass_mask)
            _append_feature(px_fft)
            del px_spc, px_fft, highpass_mask

            lx = x[:, self.high_dim:, :, :, :].contiguous()  # 低频率特征
            del x
            lx_spc = self.low_mixer(lx)
            del lx
            _append_feature(lx_spc)
            lx_fft = self.apply_frequency_mask(lx_spc, lowpass_mask)
            _append_feature(lx_fft)
            del lx_spc, lx_fft, lowpass_mask

        x_se = self.channel_attention(x_cat)  # 应用SE注意力
        x_cat.mul_(x_se)  # 逐通道缩放
        del x_se

        residual = self.relu(self.bn_fuse(self.conv_fuse(x_cat)))
        x_cat.add_(residual)  # 残差连接
        del residual

        x = self.proj(x_cat)
        del x_cat
        x = self.bn_proj(x)  # BatchNorm
        x = self.proj_drop(x)

        if visual:
            return x, features
        else:
            return x


# 测试代码
if __name__ == "__main__":
    # 创建混合器（64通道输入，128通道输出）
    mixer = HiLoBlock(in_dim=64, out_dim=128).cuda()

    # 生成测试输入 (B=1, C=64, H=32, W=32, D=32)
    input_tensor = torch.randn(2, 64, 32, 32, 32).cuda()

    # 前向传播
    output_tensor = mixer(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_tensor.shape}")  # 期望输出: (1, 128, 32, 32, 32)
