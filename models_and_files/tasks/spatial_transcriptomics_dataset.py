import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as TF
import random

class MemoryEfficientPairedLoaderWrapper: #created to
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

class MemoryEfficientPairedLoader:
    #creating pairedspaitialtranscriptomics dataset object from this memory efficient pairedloader class we load high-res and low-res .npz file pairs one at a time to handle memory constraints. each file has a dataloader instance
    #man this took too long to get working correctly
    def __init__(self, high_res_files, low_res_files, batch_size): #lists of paths to the files (files are chunked up) - Batch size is an int
        #NOTE - in the final data iteration (characteristics of the set are due to computational and time constraints) 
        self.high_res_files = high_res_files
        self.low_res_files = low_res_files
        self.batch_size = batch_size
        #There should be no error but because I am an idiot and we had an error, I will leave this here
        assert len(high_res_files) == len(low_res_files), \
            "num files mismatch between synth visium and ground truth"

     #We want to create file path pairs for corresponding synthetic visium and merfish grids - create dataloader batches as well
     #decided to set num workers to num cores-1 for safety, nvm =core count is fine - decide your own value at your own discretion
    def __iter__(self):
        for high_res_fp, low_res_fp in zip(self.high_res_files, self.low_res_files):
            dataset = PairedSpatialTranscriptomicsDataset(high_res_fp, low_res_fp)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=16  #save time via multi-threa
            )
            for batch in dataloader:
                yield batch
    def __len__(self):
        return 20 #hard-coded due to weird bug
class PairedSpatialTranscriptomicsDataset(Dataset):
    '''
    Summary: Class to create our data pairs and access our arrays/load them into memory. Note: Consistency with use of "" and '' sucks
    '''

    def __init__(self, path_high_res, path_low_res): #keeping parameter and attribute names different for ease of debugging later
        #Used to get the paths to where the files are stored (seperate high and low because in some of my preprocessed data I chose to split data storage into seperate folders) 
        #updated for single file use, created memoryefficient paired loader for lazy loading
        self.high_res_fp = path_high_res #plural paths - No longer array of strings - updated now to only be one singular path
        self.low_res_fp = path_low_res #array of strings
        self.high_res_data = None
        self.low_res_data = None
        self.num_grids = 0
        self._load_data()
        self.angles=[0,90,180,270]

    def _load_data(self):
        #the key for npz files is "grids" whereas if you are using my h5 files most have keys "low_res" and "high_res" corresponding to whichever one you want
        #the files are chunked small enough to load into memory without much issue at the moment
        self.high_res_data = np.load(self.high_res_fp)["grids"]  # Directly loads the array of grids b x c x m x m where b is 20 unless we are in the last file - c=502, m=313
        self.low_res_data = np.load(self.low_res_fp)["grids"]
        #this is a good place to throw an assertion/debug if you think something went wrong for data processing if you are using any data other than mine - dn2572
        self.num_grids = self.high_res_data.shape[0]  #should be 20 except possibly in the the last file _x for max(x) extension number
    def __len__(self): #encapsulation? I think thats the term, man I don't really do CS so meh
        return self.num_grids #use later when batching

    def __getitem__(self, idx):
        #Summary: Get a single grid from high and low res data by index. Make said grids pytorch tensors (specify
        #float 32 to avoid truncation) - create binary mask to skip padded regions in loss calcualtions
        high_res_tensor = torch.tensor(self.high_res_data[idx], dtype=torch.float32)
        low_res_tensor = torch.tensor(self.low_res_data[idx], dtype=torch.float32)
        
        #we are going to use our pre-trained interpolation model to handle this
        with torch.no_grad():  #no gradients needed, save computational power
            #add 4th dim the batch dimfor model inference
            lr_tensor = low_res_tensor.unsqueeze(0)  # [1, 502, H, W]
            #no need to generate upscaled lr
            tensor_lr_up = lr_tensor
        
            #We want to create a mask so that we don't include padded regions (value -1)  -> We can do this by creating mask for ANY single channel of said grid and then expanding said grid
            #SAME MASK APPLIES TO BOTH HIGH AND LOW RES! We designed our spatial features as such for reason
    
        mask = (high_res_tensor[-1] != -1).float()  # ANY single channel, -1 values
        mask = mask.unsqueeze(0).expand(502, -1, -1)  # Stretch over 502 channels
        rotation_angle = random.choice(self.angles) #pick an angle that is a multiple of 90 (up to 270) - no distortion 
        high_res_tensor=TF.rotate(high_res_tensor,rotation_angle) #rotate tensor pairs by same rotation angle
        low_res_tensor=TF.rotate(low_res_tensor, rotation_angle)
        return {'high_res_tensor': high_res_tensor, 
                'low_res_tensor': low_res_tensor, 
                'tensor_lr_up': tensor_lr_up, 
                'mask': mask}
class PairedSpatialTranscriptomicsDataset_2(Dataset): #FOR RRDB TRAINING - same as earlier but slightly repurposed

    def __init__(self, path_high_res, path_low_res):
        self.high_res_fp = path_high_res
        self.low_res_fp = path_low_res
        self.high_res_data = None
        self.low_res_data = None
        self.num_grids = 0
        self._load_data()

    def _load_data(self):
        self.high_res_data = np.load(self.high_res_fp)["grids"]
        self.low_res_data = np.load(self.low_res_fp)["grids"]

        # Sanity check for shape consistency
        assert self.high_res_data.shape[0] == self.low_res_data.shape[0], \
            f"Mismatch in number of grids: {self.high_res_data.shape[0]} vs {self.low_res_data.shape[0]}."
        self.num_grids = self.high_res_data.shape[0]

    def __len__(self):
        return self.num_grids

    def __getitem__(self, idx):
        
        #Fetch a high-res and low-res grid pair and prepare mask
        high_res_tensor = torch.tensor(self.high_res_data[idx], dtype=torch.float32).cuda()
        low_res_tensor = torch.tensor(self.low_res_data[idx], dtype=torch.float32).cuda()
        rotation_angle = random.choice(self.angles)
        high_res_tensor=TF.rotate(high_res_tensor,rotation_angle)
        low_res_tensor=TF.rotate(low_res_tensor, rotation_angle)
        mask = (high_res_tensor[-1] != -1).float().unsqueeze(0).expand(502, -1, -1) #expand to 502 layers directly - save time
        return {
            'high_res_tensor': high_res_tensor,
            'low_res_tensor': low_res_tensor,
            'mask': mask
        }



#The following is for RRDB Training


#The following is for RRDB Training
class MemoryEfficientPairedLoaderWrapper_2:
    """
    Wrapper for memory-efficient paired loaders (RRDB training).
    """

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class MemoryEfficientPairedLoader_2:
    """
    Memory-efficient loader for paired spatial transcriptomics datasets (RRDB training).
    """

    def __init__(self, high_res_files, low_res_files, batch_size):
        self.high_res_files = high_res_files
        self.low_res_files = low_res_files
        self.batch_size = batch_size

        assert len(high_res_files) == len(low_res_files), \
            "Mismatch in number of high-res and low-res files!"

    def __iter__(self): #iterator same as for mepl2 above
        
        for high_res_fp, low_res_fp in zip(self.high_res_files, self.low_res_files):
            dataset = PairedSpatialTranscriptomicsDataset_2(high_res_fp, low_res_fp)
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True,
                pin_memory=True, num_workers=8
            )
            for batch in dataloader:
                yield batch

    def __len__(self):
        """
        Return the total number of batches (estimated).
        """
        total_grids = sum(PairedSpatialTranscriptomicsDataset_2(fp1, fp2).__len__()
                          for fp1, fp2 in zip(self.high_res_files, self.low_res_files))
        return (total_grids + self.batch_size - 1) // self.batch_size


class PairedSpatialTranscriptomicsDataset_2(Dataset):
    def __init__(self, path_high_res, path_low_res):
        self.high_res_fp = path_high_res
        self.low_res_fp = path_low_res
        self.high_res_data = None
        self.low_res_data = None
        self.num_grids = 0
        self._load_data()
        self.angles= [0,90,180,270]
        #rotation if fixed to one of those 4 options to prevent distortions from pixel interpolation from other non pi/2 multiple angles - whatever you call them 
    def _load_data(self):
        """
        Load data from .npz files.
        """
        self.high_res_data = np.load(self.high_res_fp)["grids"]
        self.low_res_data = np.load(self.low_res_fp)["grids"]

        # Sanity check for shape consistency
        assert self.high_res_data.shape[0] == self.low_res_data.shape[0], \
            f"Mismatch in number of grids: {self.high_res_data.shape[0]} vs {self.low_res_data.shape[0]}."
        self.num_grids = self.high_res_data.shape[0]

    def __len__(self):
        """
        Return the number of grids in the dataset.
        """
        return self.num_grids

    def __getitem__(self, idx):
        #fetch's pair of tensors but with random rotations
        high_res_tensor = torch.tensor(self.high_res_data[idx], dtype=torch.float32)
        low_res_tensor = torch.tensor(self.low_res_data[idx], dtype=torch.float32)
        rotation_angle = random.choice(self.angles)
        high_res_tensor=TF.rotate(high_res_tensor,rotation_angle)
        low_res_tensor=TF.rotate(low_res_tensor, rotation_angle)
        # Create mask for valid regions
        mask = (high_res_tensor[-1] != -1).float()#mask where padded regions =0 and non-padded =1
        mask = mask.unsqueeze(0).expand(502, -1, -1)#expand mask from 2D to 3D (along all the channels)

        return {
            'high_res_tensor': high_res_tensor,
            'low_res_tensor': low_res_tensor,
            'mask': mask
        }



