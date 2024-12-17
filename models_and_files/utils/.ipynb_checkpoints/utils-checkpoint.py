import subprocess
import torch.distributed as dist
import glob
import os
import re
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

#Note we modified and re-wrote what was most relevant to us, some of the function are note re-rewritten as we do not use them but kept them from the original SRDiff file - often we just implemented our own code in the model files in place of this
def reduce_tensors(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v)
            v = v / dist.get_world_size()
        if type(v) is dict:
            v = reduce_tensors(v)
        new_metrics[k] = v
    return new_metrics


def tensors_to_scalars(tensors):
    if isinstance(tensors, torch.Tensor):
        return tensors.item()
    elif isinstance(tensors, dict):
        return {k: tensors_to_scalars(v) for k, v in tensors.items()}
    elif isinstance(tensors, list):
        return [tensors_to_scalars(v) for v in tensors]
    else:
        return tensors


def tensors_to_np(tensors):

    if isinstance(tensors, torch.Tensor):
        return tensors.cpu().numpy()
    elif isinstance(tensors, dict):
        return {k: tensors_to_np(v) for k, v in tensors.items()}
    elif isinstance(tensors, list):
        return [tensors_to_np(v) for v in tensors]
    else:
        raise TypeError(f"Unsupported type for tensors_to_np: {type(tensors)}")


def move_to_cpu(tensors):

    if isinstance(tensors, torch.Tensor):
        return tensors.cpu()
    elif isinstance(tensors, dict):
        return {k: move_to_cpu(v) for k, v in tensors.items()}
    return tensors


def move_to_cuda(batch, gpu_id=0):

    if isinstance(batch, torch.Tensor):
        return batch.cuda(gpu_id, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: move_to_cuda(v, gpu_id) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_cuda(x, gpu_id) for x in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_cuda(x, gpu_id) for x in batch)
    return batch


def get_last_checkpoint(work_dir, steps=None):

    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location='cpu')
        return checkpoint, last_ckpt_path
    return None, None


def get_all_ckpts(work_dir, steps=None):
 
    pattern = f'{work_dir}/model_ckpt_steps_{"*" if steps is None else steps}.ckpt'
    return sorted(glob.glob(pattern), key=lambda x: -int(re.findall(r'steps_(\d+)\.ckpt', x)[0]))


def load_checkpoint(model, optimizer, work_dir):

    checkpoint, _ = get_last_checkpoint(work_dir)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict']['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_states'][0])
        global_step = checkpoint['global_step']
        del checkpoint
        torch.cuda.empty_cache()
        return global_step
    return 0


def save_checkpoint(model, optimizer, work_dir, global_step, num_ckpt_keep): #for model and optimizer saving - remove old ones for space- change this if you would like to keep them
    ckpt_path = f'{work_dir}/model_ckpt_steps_{global_step}.ckpt'
    print(f'Saving checkpoint to {ckpt_path}...')
    checkpoint = {
        'global_step': global_step,
        'state_dict': {'model': model.state_dict()},
        'optimizer_states': [optimizer.state_dict()]
    }
    torch.save(checkpoint, ckpt_path)
    old_ckpts = get_all_ckpts(work_dir)[num_ckpt_keep:]
    for old_ckpt in old_ckpts:
        os.remove(old_ckpt)
        print(f"Deleted old checkpoint: {old_ckpt}")
def load_ckpt(model, ckpt_path, model_name='model', force=True, strict=True):

    #ckpt_path - believe it or not ckpt is short for checkpoint lol, how long did it take me to draw the association with .ckpt files? 
    #model_name is the name of the model in the checkpoint state_dict
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get("state_dict", checkpoint)
        if model_name in state_dict:
            state_dict = state_dict[model_name]
        if not strict:
            # Handle parameter mismatches
            current_state_dict = model.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in current_state_dict:
                    current_param = current_state_dict[key]
                    if current_param.shape != param.shape:
                        unmatched_keys.append(key)
            for key in unmatched_keys:
                del state_dict[key]
        model.load_state_dict(state_dict, strict=strict)
        print(f"| Loaded checkpoint '{model_name}' from '{ckpt_path}'.")
    else:
        if force:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"| Warning: Checkpoint not found at {ckpt_path}. Skipping load.")


class Measure:
    #adapted for spatial transcriptomics - removed irrelelvant metrics
    def __init__(self, net='alex'):
        self.model = lpips.LPIPS(net=net)

    def measure(self, prediction_hr, high_res_tensor, low_res_tensor, sr_scale,mask):
        
        #compute various metrics for high-resolution reconstruction
        #Returns dictionary of metrics: PSNR, SSIM, LPIPS.
        prediction_hr = prediction_hr
        high_res_tensor = high_res_tensor
        low_res_tensor = low_res_tensor
        return {
            'psnr': psnr(prediction_hr*mask, high_res_tensor*mask, data_range=10),
            'ssim': ssim(prediction_hr*mask, high_res_tensor*mask, multichannel=True, data_range=10),
            'lpips': self.lpips(prediction_hr, high_res_tensor, mask)
        }

    def lpips(self, high_res_tensor,low_res_tensor,prediction_hr, mask):
        return self.model.forward(high_res_tensor,low_res_tensor, prediction,mask).item()


def remove_file(*fns):
    #some utility functionality from the referenced SRDiff code I decided to keep
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)
def load_ckpt(model, ckpt_path, model_name="model", force=False, strict=True):
    #Load the checkpoint for the model, extracting the model_state_dict.
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Extract the model state dictionary
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict'. Keys: {checkpoint.keys()}")

    # Remove unmatched keys if force is True
    if force:
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        unmatched_keys = ckpt_keys - model_keys
        for key in unmatched_keys:
            del state_dict[key]

    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=strict)
    print(f"| Loaded checkpoint '{model_name}' from '{ckpt_path}'.")
