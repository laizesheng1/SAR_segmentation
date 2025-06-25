import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class EfficientMSA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x: [B, N, C]
        attn_out, _ = self.attn(x, x, x)
        return attn_out
#input=output=[B C H W]
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super(TransformerBlock, self).__init__()

        # Depth-wise convolution (DW Conv)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        # Layer Norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Efficient Multi-head Self-Attention (EMSA) placeholder
        self.attn = EfficientMSA(dim, num_heads)

        # Residual Feed-forward network
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # DW Conv
        x = self.dw_conv(x)
        # Flatten for transformer: [B, C, H, W] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        # Attention block
        x = x + self.attn(self.norm1(x))
        # FFN block
        x = x + self.ffn(self.norm2(x))
        # Reshape back to image format
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x



