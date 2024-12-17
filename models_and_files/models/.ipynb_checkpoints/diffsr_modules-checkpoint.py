from torchvision.models import resnet18
import torch
from torch import nn
import torch.nn.functional as F
from utils.hparams import hparams
from .module_util import make_layer, initialize_weights
from .commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from .commons import ResnetBlock, Upsample, Block, Downsample
import functools
import torch
from torch import nn
import torch.nn.functional as F
from utils.hparams import hparams
from .module_util import make_layer, initialize_weights
from .commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from .commons import ResnetBlock, Upsample, Block, Downsample

#Residual Dense Block (RRDBNet) for conditional feature extraction
class RRDBNet(nn.Module):
    """
    Residual in Residual Dense Network (RRDBNet) for feature extraction and upscaling.
    Used as a conditional input to the main diffusion model. num_feature_maps=32, num_blocks=6, growth_channels=32
    """
    def __init__(self, in_nc=502, out_nc=502,num_feature_maps=64, num_blocks=8, growth_channels=32):

        #Based on observed standards ex: ERGAN, ESRGAN, 64 to 128 num_feature_maps is recommended - 64 if lighter arch
        #growth_channels - growth channels - more channels if you have more complex channel dependencies (between genes)

        super(RRDBNet, self).__init__()
        #Super cool feature utilize functools.partial to create a preconum_feature_mapsigured version of the RRDB class!!
        RRDB_block_f = functools.partial(RRDB, num_feature_maps=num_feature_maps, growth_channels=growth_channels)

        #start first convolution layer
        self.conv_first = nn.Conv2d(in_nc, num_feature_maps, 3, 1, 1, bias=True)

        #residual dense block trunk
        self.RRDB_trunk = make_layer(RRDB_block_f, num_blocks)

        #conv is applied after the trunk
        self.trunk_conv = nn.Conv2d(num_feature_maps, num_feature_maps, 3, 1, 1, bias=True)

        #res refinement layers
        self.upconv1 = nn.Conv2d(num_feature_maps, num_feature_maps, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feature_maps, num_feature_maps, 3, 1, 1, bias=True)

        #final high-resolution output layers
        self.HRconv = nn.Conv2d(num_feature_maps, num_feature_maps, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_feature_maps, out_nc, 3, 1, 1, bias=True)

        #activation - leaky relu but also try MISH
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, get_fea=False):
        """
        Forward pass for RRDBNet.

        Args:
            x (Tensor): Input tensor with shape (batch_size, 502, height, width).
            get_fea (bool): If True, returns intermediate features.

        Returns:
            Tensor: Output tensor with high-resolution features.
        """
        # Check input channel consistency for spatial transcriptomics
        if len(x.shape)==3:
            x=x.unsqueeze(0)
        if x.shape[1] != 502:
            raise ValueError(f"Expected input channels to be 502, got {x.shape[1]}")

        #no normalization - by default log 2 normalized but you can normalize here if desired for your data
        #x = x * 2 - 1

        # Store intermediate features if needed
        feas = []
        fea_first = fea = self.conv_first(x)
        for l in self.RRDB_trunk:
            fea = l(fea)
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        feas.append(fea)

        #We do not need to upsample but if you want to for your own dataset, then swap out the uncommented upconv1           and 2 with interpolation ,layers. Also adjust scale factor and add more layers for sr>4 and a factor of 2
        #fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        #fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # Resolution refinement (no spatial upscaling)
        fea = self.lrelu(self.upconv1(fea))#refinement layer without upscaling
        fea = self.lrelu(self.upconv2(fea))#another refinement layer
        fea_hr = self.HRconv(fea)
        # Generate final output with clamping to [-1, 1]
        out = self.conv_last(self.lrelu(fea_hr))
        out = out.clamp(-1, 1)

        return (out, feas) if get_fea else out



class Unet(nn.Module):
    def __init__(self, dim, out_dim=502, dim_mults=(1, 2, 4, 8), cond_dim=5):
        super().__init__()
        dims = [502, *map(lambda m: dim * m, dim_mults), 502]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"Unet dims: {dims}")
        self.lr_up_proj = nn.Conv2d(502, 64, kernel_size=1, stride=1)
        groups = hparams.get('groups', 1)
        self.cond_proj = nn.ConvTranspose2d(
            cond_dim * ((hparams['rrdb_num_block'] + 1) // 3), 502,
            hparams['sr_scale'] * 2, hparams['sr_scale'], hparams['sr_scale'] // 2
        )
        self.mid_block1 = ResnetBlock(dims[-1], dims[-1], time_emb_dim=dim, groups=groups)
        self.mid_block2 = ResnetBlock(dims[-1], dims[-1], time_emb_dim=dim, groups=groups)

        if hparams['use_attn']:
            self.mid_attn = LinearAttention(dim=dims[-1])
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim))
        self.downs = nn.ModuleList([
            self._build_down(dim_in, dim_out, dim, groups, is_last)
            for dim_in, dim_out, is_last in zip(dims[:-1], dims[1:], [False] * (len(dims) - 2) + [True])
        ])
        self.ups = nn.ModuleList([
            self._build_up(dim_out * 2, dim_in, dim, groups, is_last)
            for dim_in, dim_out, is_last in zip(dims[1:], reversed(dims[:-1]), [False] * (len(dims) - 2) + [True])
        ])
        self.final_conv = nn.Sequential(Block(dim, dim, groups=groups), nn.Conv2d(dim, out_dim, 1))

    def _build_down(self, dim_in, dim_out, time_emb_dim, groups, is_last):
        return nn.ModuleList([
            ResnetBlock(dim_in, dim_out, time_emb_dim=time_emb_dim, groups=groups),
            ResnetBlock(dim_out, dim_out, time_emb_dim=time_emb_dim, groups=groups),
            Downsample(dim_out) if not is_last else nn.Identity()
        ])

    def _build_up(self, dim_in, dim_out, time_emb_dim, groups, is_last):
        return nn.ModuleList([
            ResnetBlock(dim_in, dim_out, time_emb_dim=time_emb_dim, groups=groups),
            ResnetBlock(dim_out, dim_out, time_emb_dim=time_emb_dim, groups=groups),
            Upsample(dim_out) if not is_last else nn.Identity()
        ])

    def forward(self, x, time, cond, tens_lr_up):
        print(f"Unet input: {x.shape}")
        t = self.time_pos_emb(time)
        t = self.mlp(t)
        print(f"Time embedding: {t.shape}")
        cond = self.cond_proj(cond)
        print(f"Cond projected: {cond.shape}")
        h = []
        for i, (resnet1, resnet2, downsample) in enumerate(self.downs):
            x = resnet1(x, t)
            x = resnet2(x, t)
            if i == 0:
                x = x + self.lr_up_proj(tens_lr_up)
            print(f"Downsample {i}: {x.shape}")
            h.append(x)
            x = downsample(x)
        print(f"Mid-block input: {x.shape}")
        x = self.mid_block1(x, t)
        if hparams['use_attn']:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for i, (resnet1, resnet2, upsample) in enumerate(self.ups):
            skip = h.pop()
            x = torch.cat((x, skip), dim=1)
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = upsample(x)
            print(f"Upsample {i}: {x.shape}")
        print(f"Unet output: {x.shape}")
        return self.final_conv(x)
#This was a feature to remove weight normalization to speed up generation from the original paper. I do not use it but you can
    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(remove_weight_norm)

class resnet_18_based_mod(nn.Module):
    def __init__(self):
        super(resnet_18_based_mod, self).__init__()
        # Encoder: Use ResNet18 for feature extraction
        self.encoder = resnet18(weights=None)  # No pretrained weights
        self.encoder.conv1 = nn.Conv2d(502, 64, kernel_size=5, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()  #getting rid of the fully connected layer due to feature extraction purposes
        #here we implement a decoder - aiming to get the dimensions roughly where we want them to minimize upsampling at the end which will dillute our feature extraction efforts
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),   
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=1),   
            nn.ReLU(),
            nn.ConvTranspose2d(32, 502, kernel_size=2, stride=1, padding=1), 
            nn.Upsample(size=(313, 313), mode='bilinear', align_corners=True)  #upsampling extracted featuers is not great as it will mess up our encoded latent feature information - intentionally choosing it to be similar dimensionally to our hr/lr tensors
        )
        self.to(torch.device("cuda"))
    def forward(self, x):
        features = self.encoder(x)  # Extract features
        features = features.view(features.size(0), -1, 1, 1)  # Reshape for decoding
        output = self.decoder(features)  # Decode to high-resolution size
        return output





