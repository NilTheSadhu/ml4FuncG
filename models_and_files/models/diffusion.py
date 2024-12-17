import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from models.module_util import default
from utils.sr_utils import SSIM
from utils.hparams import hparams
from models.new_noise_scheduler import NoiseScheduler  # Import the new scheduler
from models.diffsr_modules import resnet_18_based_mod
from utils.utils import load_ckpt

def noise_like(shape, device, repeat=False):
    #Generate random noise tensors with the specified shape.
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, rrdb_net=None, timesteps=5, loss_type='l1', noise_scheduler=None):

        #Integrated NoiseScheduler for custom noise scheduling.
        #Adjusted for 502-channel spatial transcriptomics data.
        #RRDBNet used for conditional generation with low-resolution grids.

        super().__init__()
        self.denoise_fn = denoise_fn
        if rrdb_net:
            self.rrdb = rrdb_net  #conditions our noising 
            self.rrdb_net=self.rrdb #create reference for lazy debug
            if hparams['rrdb_ckpt'] != '' and  hparams['use_rrdb']:
                load_ckpt(self.rrdb, hparams['rrdb_ckpt'])
        else:
            device = torch.device('cuda')
            # Load ResNet-based feature extractor
            model_resnet = resnet_18_based_mod()
            try:
                model_resnet.load_state_dict(torch.load("./interpolation_model.pth", map_location=device)) #excuse the misleading file name - it started off interpolating before feature extracting via cnn layers
            except RuntimeError as e:
                print("error loading the (poorly named) lnterpolation_Model.pth: ", e)
                raise
            model_resnet.to(device)
            model_resnet.eval()
            self.rrdb = model_resnet
        device = torch.device('cuda')
        self.noise_scheduler = noise_scheduler or NoiseScheduler(
            num_timesteps=timesteps,
            beta_schedule='linear',
            beta_start=0.0001,
            beta_end=0.02,
            device=device
        )
        self.num_timesteps = timesteps
            
    def forward(self, high_res, low_res, low_res_up, mask, t=None, *args, **kwargs):
        
        b, *_, device = *high_res.shape, high_res.device #should be cuda
        if t is None: #usually will be none since our scheduler class already has it
            t = torch.randint(0, self.noise_scheduler.num_timesteps, (b,), device=device).long()

    
        #This is key to the SRDIFF strategy where we look at the residuals residual: HR - upsampled LR
        residual = high_res - low_res_up
        
        # Low-resolution guidance from RRDB or ResNet-based model
        if hparams['use_rrdb']:
            rrdb_out, cond = self.rrdb.forward(low_res, True)  #using thee rrdb net -> rrdbout is the upscaled - cond is what we are conditioning on
        else:
            rrdb_out = low_res_up #low_res_up=low_res for our current given data structure
            cond = self.rrdb.forward(low_res)  #extracted fea to condition on
    
        #masking matrices before loss calculations to exclude padded regions (-1 padding for uniform size)
        residual_masked = residual * mask
        loss, x_tp1, noise_pred = self.p_losses(residual_masked, t, cond, low_res_up * mask, *args, **kwargs)
    
        #added losses
        ret = {'q': loss}
        if hparams['aux_l1_loss']:
            ret['aux_l1'] = F.l1_loss(rrdb_out, residual)  # Compute L1 loss on residuals
        if hparams['aux_ssim_loss']:
            ret['aux_ssim'] = 1 - self.ssim_loss(rrdb_out, residual)
    
        return ret, (x_tp1, noise_pred), t

    def p_losses(self, residual, t, cond, low_res_up, noise=None):
        #compute prediction losses using the specified diffusion process on the residual.
        noise = default(noise, lambda: torch.randn_like(residual))
        x_tp1_gt = self.noise_scheduler.add_noise(residual, noise, t)  # Add noise to the residual
        noise_pred = self.denoise_fn(x_tp1_gt, t, cond, low_res_up)  # Predict noise
    
        #loss time initial options guassian
        if self.loss_type == 'l1':
            loss = (noise - noise_pred).abs().mean()  
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, noise_pred)  
        elif self.loss_type == 'ssim':
            loss = (1 - self.ssim_loss(noise, noise_pred))  
        else:
            raise NotImplementedError()
    
        return loss, x_tp1_gt, noise_pred

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the input with noise at a given time step using the NoiseScheduler.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return self.noise_scheduler.add_noise(x_start, noise, t)
    def p_sample(self, x_t, t, model_output, low_res_up):
        """
        Reverse diffusion step: Predict residual and reconstruct SR image.
        """
        residual_pred, _ = self.noise_scheduler.step(
            model_output=model_output,
            timestep=t,
            sample=x_t,
            model_pred_type="noise"
        )
        sr_image = residual_pred + low_res_up  #predicted residual + upsampled LR image to form the final image
        return sr_image


