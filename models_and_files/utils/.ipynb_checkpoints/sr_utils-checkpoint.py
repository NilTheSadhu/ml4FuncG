import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
from math import exp
import torch.nn as nn


##########
#We do not use this per say but we kept the original SRDiff implements, see code for our own unique implements. If you do want to use this, we modified enough of the code to make it compliant with our data
##########
def gaussian(window_size, sigma):
    """Generate a Gaussian kernel."""
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):#channel by channel
    """Create a 2D Gaussian window."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(tensor1, tensor2, window, window_size, channel, size_average=True, data_range=9.7):#data range is based on exploratory data analysis of our data
    tensor1 = tensor1 / data_range
    tensor2 = tensor2 / data_range
    mu1 = F.conv2d(tensor1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(tensor2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(tensor1 * tensor1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(tensor2 * tensor2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(tensor1 * tensor2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    #values based on numbers from both SRDiff and STDiff (later focuses on transcriptomics data - albeit intergrating multiple omics)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
#SSIM module for multi-channel data (e.g., spatial transcriptomics).
    def __init__(self, window_size=10, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 502  # Updated for 502 channels
        self.window = create_window(window_size, self.channel)

    def forward(self, tensor1, tensor2):
        #batches by channels by  x and y
        tensor1 = tensor1 * 0.5 + 0.5
        tensor2 = tensor2 * 0.5 + 0.5
        (_, channel, _, _) = tensor1.size()

        if channel == self.channel and self.window.data.type() == tensor1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if tensor1.is_cuda:
                window = window.cuda(tensor1.get_device())
            window = window.type_as(tensor1)

            self.window = window
            self.channel = channel

        return _ssim(tensor1, tensor2, window, self.window_size, channel, self.size_average)



