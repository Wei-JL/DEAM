import math
import torch
import torch.nn as nn

from transfromer_encode import TransformerBlock


class ChannelAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ChannelAttention, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conva = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.convm = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        # self.silu = nn.SiLU()
        self.silu = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x.flatten(2)
        ya = self.avg_pool(x)
        ya = self.conva(ya.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)

        ym = self.max_pool(x)
        ym = self.convm(ym.squeeze(-1).transpose(-1, -2))

        y = torch.matmul(ya, ym)  # 矩阵相乘尺寸c*c

        out = torch.matmul(y, residual).view(B, C, H, W)

        out = self.silu(out)
        return out


class SpatialSelfAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(SpatialSelfAttention, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.conv = nn.Conv2d(2, channel, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.transfromer = TransformerBlock(dim=channel)
        self.conv_out = nn.Conv2d(channel, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.transfromer(x)
        x = self.conv_out(x)
        x = x * residual
        return self.silu(x)


if __name__ == '__main__':
    x = torch.randn([2, 16, 256, 256])
    model_c = ChannelAttention(16)
    model_s = SpatialSelfAttention(16)

    out_s = model_s(x)
    out_c = model_c(x)

    print(out_c)
