import os
import torch
from torch.utils.data import DataLoader
from models.diffusion import GaussianDiffusion
from models.diffsr_modules import RRDBNet, Unet, resnet_18_based_mod
from models.new_noise_scheduler import NoiseScheduler
from utils.hparams import hparams
from utils.utils import Measure, move_to_cuda, load_checkpoint, save_checkpoint, tensors_to_scalars
from tqdm import tqdm
from tasks.spatial_transcriptomics_dataset import MemoryEfficientPairedLoader, PairedSpatialTranscriptomicsDataset,MemoryEfficientPairedLoaderWrapper
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
#to help solve bottleneck mystery
def monitor_gpu_usage():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0) #gpu 0, adjust as needed

    # Get memory and utilization info
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    mem_total = mem_info.total/ (1024** 3)  
    mem_used = mem_info.used/(1024** 3)   
    mem_free = mem_info.free/(1024** 3)   
    utilization = nvmlDeviceGetUtilizationRates(handle)
    gpu_util = utilization.gpu  #percentage

    print(f"GPU Usage:")
    print(f"Memory Total: {mem_total:.2f} GB")
    print(f"Memory Used: {mem_used:.2f} GB")
    print(f"Memory Free: {mem_free:.2f} GB")
    print(f"GPU Util: {gpu_util}%")

monitor_gpu_usage()

class SRDiffTrainer:
    """
    Trainer class for the SRDiff task, tailored for spatial transcriptomics.
    """
    def __init__(self, train_data_dir, val_data_path, test_data_path):
        """
        Initialize the trainer with paths, metrics, and logging setup.
        """
        self.train_data_dir = train_data_dir
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.model = None  # Will be initialized in `build_model`
        self.logger = self.build_tensorboard(hparams['work_dir'], 'tb_logs')
        self.measure = Measure() 
        self.metric_keys = ['psnr', 'mse', 'ssim']  # Metrics for evaluation
        self.work_dir = hparams['work_dir']
        self.global_step = 0
        self.first_val = True  # Flag for limited validation during the first epoch
        self.rrdb=None
    def build_tensorboard(self, save_dir, name):
        """
        Creates a TensorBoard logger.
        """
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir)

    def build_model(self):
        """
        Build the SRDiff model using GaussianDiffusion.
        """
        noise_scheduler = None #let the initialization for our NS object happen in the diffusion model code
        self.model = GaussianDiffusion(
            denoise_fn=Unet(
                dim=hparams['hidden_size'],
                out_dim=502,
                cond_dim=502 #cond_dim=hparams['rrdb_num_feat'] #Uncomment if using rrdb,otherwise use preset 502
                ,dim_mults=[int(x) for x in hparams['unet_dim_mults'].split('|')],
            ),
            rrdb_net=RRDBNet(
                in_nc=502, out_nc=502, num_feature_maps=hparams['rrdb_num_feat'],
                num_blocks=hparams['rrdb_num_block'], growth_channels=hparams['rrdb_num_feat'] // 2
            ) if hparams['use_rrdb'] else None, #in the case of none we build our alternative model when None is passed
            timesteps=hparams['timesteps'],
            loss_type=hparams['loss_type'],
            noise_scheduler=noise_scheduler
        )
        self.model.to(torch.device("cuda"))
        return self.model

    def build_train_dataloader(self):
        """
        Construct memory-efficient dataloader for training.
        """
        high_res_files = [f"{self.train_data_dir}/high_res_grids_{x}.npz" for x in range(1, 4)]
        low_res_files = [f"{self.train_data_dir}/low_res_grids_{x}.npz" for x in range(1, 4)]
        return MemoryEfficientPairedLoader(high_res_files, low_res_files, hparams['batch_size']) #I hard coded batch values because I was lazy, feel free to change it back to hparams["batch_size"]

    def build_val_dataloader(self):
        """
        Construct validation dataloader.
        """
        high_res_files = [f"{self.train_data_dir}/high_res_grids_{x}.npz" for x in range(4, 5)]
        low_res_files = [f"{self.train_data_dir}/low_res_grids_{x}.npz" for x in range(4, 5)]
        return MemoryEfficientPairedLoader(high_res_files, low_res_files,batch_size=hparams['eval_batch_size'])

    def build_test_dataloader(self):
        """
        Construct test dataloader.
        """
        high_res_files = [f"{self.train_data_dir}/high_res_grids_{x}.npz" for x in range(5, 6)]
        low_res_files = [f"{self.train_data_dir}/low_res_grids_{x}.npz" for x in range(5, 6)]
        return MemoryEfficientPairedLoader(high_res_files, low_res_files,batch_size=hparams['eval_batch_size'])

    def build_optimizer(self, model):
        """
        Define optimizer, optionally excluding RRDB parameters.
        """
        params = list(model.named_parameters())
        if hparams.get('fix_rrdb', False):
            params = [p for p in params if 'rrdb' not in p[0]]
        params = [p[1] for p in params]
        return torch.optim.Adam(params, lr=hparams['lr'])

    def build_scheduler(self, optimizer):
        """
        Define learning rate scheduler.
        """
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=hparams.get('decay_steps', 10000),
            gamma=hparams.get('decay_gamma', 0.5)
        )

    def training_step(self, sample):
        img_hr = sample['high_res_tensor']
        img_lr = sample['low_res_tensor']
        mask = sample['mask']
        predictions = self.model(img_lr)
        loss = hybrid_ssim_mse(predictions * mask, img_hr * mask)  #use default weight and alpha/beta
        return {'l': loss, 'lr': self.scheduler.get_last_lr()[0]}, loss


    def train(self):
        #main training loop
        self.model = self.build_model()
        optimizer = self.build_optimizer(self.model)
        self.global_step = load_checkpoint(self.model, optimizer, self.work_dir)
        scheduler = self.build_scheduler(optimizer)
        monitor_gpu_usage()
        train_dataloader = self.build_train_dataloader()

        train_pbar = tqdm(train_dataloader, initial=self.global_step, total=float('inf'), dynamic_ncols=True, unit='step')
        monitor_gpu_usage()
        while self.global_step < hparams['max_updates']:
            for batch in train_pbar:
                if self.global_step % hparams['val_check_interval'] == 0:
                    self.validate(self.global_step)
                    save_checkpoint(self.model, optimizer, self.work_dir, self.global_step, hparams['num_ckpt_keep'])

                self.model.train()
                batch = move_to_cuda(batch)
                losses, total_loss = self.training_step(batch)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                self.global_step += 1
                scheduler.step(self.global_step)

                if self.global_step % 100 == 0:
                    self.log_metrics({f'tr/{k}': v for k, v in losses.items()}, self.global_step)
                train_pbar.set_postfix(**tensors_to_scalars(losses))

    def validate(self, step):
        #validation loop and standard computations
        val_dataloader = self.build_val_dataloader()
        total_samples = len(val_dataloader) #hard coded in the object itself due to bug
        pbar = tqdm(val_dataloader, total=total_samples)
        for batch_idx, batch in enumerate(pbar):
            if self.first_val and batch_idx > hparams['num_sanity_val_steps']:
                break
            batch = move_to_cuda(batch) #load on gpu 
            _, metrics = self.sample_and_test(batch)
            pbar.set_postfix(**metrics)

    def test(self):
        #rebuild model and load the prior information from training
        self.model = self.build_model()
        optimizer = self.build_optimizer(self.model)
        load_checkpoint(self.model, optimizer, self.work_dir)
        #building tesloader
        test_dataloader = self.build_test_dataloader()

        results = {k: 0 for k in self.metric_keys}
        results['n_samples'] = 0

        with torch.no_grad():
            self.model.eval()
            for batch in test_dataloader:
                batch = move_to_cuda(batch)
                _, metrics = self.sample_and_test(batch)

                for k in self.metric_keys:
                    results[k] += metrics[k]
                results['n_samples'] += 1

        results = {k: v / results['n_samples'] for k, v in results.items()}
        print(f"Test results: {results}")

    def sample_and_test(self, sample):
        """
        Sampling and metric evaluation
        """
        low_res_sample = sample['low_res_tensor']
        high_res_sample = sample['high_res_tensor']
        mask = sample['mask']
        
        #make sure everything is on the same device
        device = torch.device("cuda")
        high_res_sample = high_res_sample.to(device)
        low_res_sample = low_res_sample.to(device)
        mask = mask.to(device)
        rrdb_out=low_res_sample
        if hparams['use_rrdb']: #Use the trained encoder rather than the Bayes Space or our autoencoder
            rrdb_out, cond = self.model.rrdb_net.forward(low_res_sample, True)
        else: 
            tensor_lr_up = low_res_sample #We do not need to interpolate as we already have the same dims here
            rddb_out=tensor_lr_up
            
        with torch.no_grad(): #disabling gradients since we don't need to do a backwards pass and update anything
            tensor_sr = self.model.forward(high_res_sample,low_res_sample, rddb_out,mask)
        
        metrics = {} #remove to save time
        if 'psnr' in hparams['metrics']:
            metrics['psnr'] = self.measure.psnr(tensor_sr * mask, high_res_sample * mask)
        if 'ssim' in hparams['metrics']:
            metrics['ssim'] = self.measure.ssim(tensor_sr * mask, high_res_sample * mask)
        if 'lpips' in hparams['metrics']:
            metrics['lpips'] = self.measure.lpips(tensor_sr*mask, high_res_sample*mask)
        return tensor_sr, metrics
def hybrid_ssim_mse(grid1, grid2, alpha=0.2, beta=0.8, weight_factor=30):
    
    #alpha is proportion contribution ssim
    #beta is proportion contributweighted MSE 
    #weight_factor is a Scaling factor for the penalty on the last two channels in the MSE value

    # Ensure the grids have the same shape
    assert grid1.shape == grid2.shape, "Grid shapes must match!"
    
    # Compute SSIM (averaged over all channels if multi-channel)
    ssim_score = ssim(grid1, grid2, data_range=grid2.max() - grid2.min(), multichannel=True)
    
    # Compute weighted MSE
    mse = (grid1 - grid2) ** 2
    weights = np.ones_like(grid1)
    weights[:, :, -2:] *= weight_factor  # Apply heavier weights to the last two channels
    weighted_mse = np.mean(mse * weights)
    
    # Combine SSIM and Weighted MSE
    hybrid_metric = alpha * (1 - ssim_score) + beta * weighted_mse
    return hybrid_metric
    
    # Combine SSIM and Weighted MSE
    hybrid_metric = alpha * (1 - ssim_score) + beta * weighted_mse
    return hybrid_metric





