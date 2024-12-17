import torch
from torch.nn import functional as F
import numpy as np
import math

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """ Beta schedule
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class NoiseScheduler:
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear",device=torch.device('cuda')):
        """
        Initialize the NoiseScheduler with a specified beta schedule.
        Supports linear, quadratic, and cosine schedules.
        """

        device = device

        self.num_timesteps = num_timesteps
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == 'cosine':
            self.betas = torch.from_numpy(
                betas_for_alpha_bar(
                    num_timesteps,
                    lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
                ).astype(np.float32)
            ).to(device)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.).to(device)

        # Precompute frequently used terms
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod).to(device)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1).to(device)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).to(device)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod).to(device)

    def reconstruct_x0(self, x_t, t, noise):
        """
        Reconstruct x_0 (clean sample) from x_t and predicted noise.
        """
        t = t.view(-1)  # Ensure t is 1D
        s1 = self.sqrt_inv_alphas_cumprod[t].view(-1, 1, 1, 1)  # Reshape for broadcasting
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].view(-1, 1, 1, 1)  # Reshape for broadcasting
        x0 = s1 * x_t - s2 * noise
        return torch.clamp(x0, min=-1, max=1)

    def q_posterior(self, x_0, x_t, t):
        """
        Compute the posterior mean x_t-1 using x_0 and x_t.
        """
        t = t.view(-1)  # Ensure t is 1D
        s1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1)  # Reshape for broadcasting
        s2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1)  # Reshape for broadcasting
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        """
        Get the variance for the reverse diffusion step at time t.
        """
        if t == 0:
            return torch.tensor(0.0, device=self.betas.device)

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance.to(t.device)

    def step(self, model_output, timestep, sample, model_pred_type='noise'):
        """
        Perform a reverse diffusion step.
        Args:
            model_output: Predicted noise or x_0 from the model.
            timestep: Current diffusion timestep (t).
            sample: Current noisy sample (x_t).
            model_pred_type: Type of prediction ('noise' or 'x_start').
        Returns:
            x_t-1 (previous timestep sample), predicted x_0.
        """
        t = timestep.view(-1)  # Ensure t is 1D
        if model_pred_type == 'noise':
            pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        elif model_pred_type == 'x_start':
            pred_original_sample = model_output
        else:
            raise NotImplementedError()

        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)  # Compute x_t-1 mean

        variance = 0
        if t.max() > 0:  # Add stochasticity for t > 0
            noise = torch.randn_like(model_output)
            variance = self.get_variance(t).view(-1, 1, 1, 1) ** 0.5 * noise

        pred_prev_sample = pred_prev_sample + variance  # Reparameterization
        return pred_prev_sample, pred_original_sample

    def add_noise(self, x_start, x_noise, timesteps):
        """
        Forward diffusion: Add noise to x_0 to produce x_t.
        Args:
            x_start: Original clean sample (x_0).
            x_noise: Random noise.
            timesteps: Diffusion timesteps (t).
        Returns:
            Noisy sample (x_t).
        """
        device=torch.device("cuda")
        s1 = self.sqrt_alphas_cumprod[timesteps].to(device).view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].to(device).view(-1, 1, 1, 1)
        return s1 * x_start + s2 * x_noise

    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        """
        Debug utility for undoing a reverse step.
        """
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        beta = self.betas[t]
        img_in_est = torch.sqrt(1 - beta) * img_out + \
                     torch.sqrt(beta) * torch.randn_like(img_out)
        return img_in_est

    def __len__(self):
        """
        Return the number of diffusion timesteps.
        """
        return self.num_timesteps
