import math
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
#residual module from original
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


#sinusoidal positional embedding - adapted from SRDiff code with minimal chagnes
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


#wide_mish activation function is also a function. If you are just using mish then you can call F.mish or whatever, no need to call this
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


#rezero module for controlling residual connections - from original
class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g



#block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim_out, 3),
            nn.GroupNorm(groups, dim_out) if groups > 0 else nn.Identity(),
            Mish()
        )

    def forward(self, x):
        return self.block(x)


#resNet Block updated to work with spatial transcriptomics
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=0, groups=8):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out)) if time_emb_dim > 0 else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        h = self.block1(x)
        if time_emb is not None and self.mlp:
            h += self.mlp(time_emb)[:, :, None, None]
        if cond is not None:
            h += cond
        h = self.block2(h)
        return h + self.res_conv(x)


#upsample module
class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
    def forward(self, x):
        return self.conv(x)


# Downsample module
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 2)
        )

    def forward(self, x):
        return self.conv(x)


#linear Attention updated for high-dimensional data with channels (500 +2)
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (heads c) h w -> b heads c (h w)', heads=self.heads), qkv)

        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


#redesigned block for our ST application
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, num_feature_maps=64, growth_channels=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feature_maps, growth_channels, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(num_feature_maps + growth_channels, growth_channels, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(num_feature_maps + 2 * growth_channels, growth_channels, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(num_feature_maps + 3 * growth_channels, growth_channels, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(num_feature_maps + 4 * growth_channels, num_feature_maps, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        #print(f"[ResidualDenseBlock_5C] Shape of x: {x.shape}")
        #print(f"[ResidualDenseBlock_5C] Shape of x1: {x1.shape}")
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


#Residual in Residual Dense Block
class RRDB(nn.Module):
    def __init__(self, num_feature_maps, growth_channels=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(num_feature_maps, growth_channels)
        self.RDB2 = ResidualDenseBlock_5C(num_feature_maps, growth_channels)
        self.RDB3 = ResidualDenseBlock_5C(num_feature_maps, growth_channels)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

