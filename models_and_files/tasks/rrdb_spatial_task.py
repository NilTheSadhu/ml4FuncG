import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from utils.hparams import hparams
from models.diffsr_modules import RRDBNet
from tasks.spatial_transcriptomics_dataset import (
    MemoryEfficientPairedLoader_2,
    MemoryEfficientPairedLoaderWrapper_2,
    PairedSpatialTranscriptomicsDataset_2
) #did you know you could do this, I didn't till this semester 0_o
from skimage.metrics import structural_similarity as ssim


class RRDBSpatialTask:
    #Task class for training the RRDB model on spatial transcriptomics data. Lots of classes, i've never really used      #classes until this class. Gotta love making classes for class
    def __init__(self, train_data_dir, val_data_dir, test_data_dir, batch_size=20): #same logic as in main
        #SRDiffTrainer file - view it for logic explanation
        '''
        train_high_res_files = [f"binned_data/high_res_grids_{i}.npz" for i in [4,8,12,13,16,18,22]]
        train_low_res_files = [f"binned_data/low_res_grids_{i}.npz" for i in [4,8,12,13,16,18,22]] 
        val_high_res_files = [f"binned_data/high_res_grids_{i}.npz" for i in [9,11,24]]
        val_low_res_files = [f"binned_data/low_res_grids_{i}.npz" for i in [9,11,24]]
        test_high_res_files = [f"binned_data/high_res_grids_{i}.npz" for i in [27,2]]
        test_low_res_files = [f"binned_data/low_res_grids_{i}.npz" for i in [27,2]]
        '''
        #choosing diverse samples, but fewer samples due to runtime constraints
        train_high_res_files = [f"binned_data/high_res_grids_{i}.npz" for i in [4,8,16,22]]
        train_low_res_files = [f"binned_data/low_res_grids_{i}.npz" for i in [4,8,16,22]] 
        val_high_res_files = [f"binned_data/high_res_grids_{i}.npz" for i in [9,11]]
        val_low_res_files = [f"binned_data/low_res_grids_{i}.npz" for i in [9,11]]
        test_high_res_files = [f"binned_data/high_res_grids_{i}.npz" for i in [27,2]]
        test_low_res_files = [f"binned_data/low_res_grids_{i}.npz" for i in [27,2]]

        #setting up our loaders with the new and improved wrappers! Wooohooo
        self.train_loader = MemoryEfficientPairedLoaderWrapper_2(
            MemoryEfficientPairedLoader_2(
                high_res_files=train_high_res_files,
                low_res_files=train_low_res_files,
                batch_size=batch_size
            )
        )
        self.val_loader = MemoryEfficientPairedLoaderWrapper_2(
            MemoryEfficientPairedLoader_2(
                high_res_files=val_high_res_files,
                low_res_files=val_low_res_files,
                batch_size=batch_size
            )
        )
        self.test_loader = MemoryEfficientPairedLoaderWrapper_2(
            MemoryEfficientPairedLoader_2(
                high_res_files=test_high_res_files,
                low_res_files=test_low_res_files,
                batch_size=batch_size
            )
        )

        #key model dims 0 hidden size must be >=2 otherwise you'll get problems with gc=0
        #hidden_size = 2 #for initial debugging set 2
        #num_blocks = 1 #for initial debugging = do not hardcode set 1  
        self.model = RRDBNet() #submit hparams if you want but I set the "best guess" params as defaults in function

        # Optimizer and Scheduler
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.5)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def move_to_device(self, batch):
        """
        Move batch data to the appropriate device.

        Args:
            batch (dict): Dictionary containing tensors for high-res, low-res grids and masks.

        Returns:
            dict: Dictionary with tensors moved to the correct device.
        """
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
    def save_checkpoint(epoch, model, optimizer, train_losses, val_losses, checkpoint_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, checkpoint_path)
    def training_step(self, sample):
        tensor_hr = sample['high_res_tensor']
        tensor_lr = sample['low_res_tensor']
        mask = sample['mask']
        print(tensor_lr.shape)
        predictions = self.model(tensor_lr)
        #they shouldn't be notably mismatched but this is for slight adjustments due to odd dimensions considering
        #we have a odd number for m - bad padding choice.
        if predictions.shape[2:] != tensor_hr.shape[2:]: #minor shape alignment only - debug if you change structure
            #don't rely heavily on reshaping or it kills the purpose of this
            predictions = F.interpolate(predictions, size=(313,313), mode='bilinear', align_corners=False)
        loss = hybrid_ssim_mse(predictions * mask, tensor_hr * mask)  # Weighted loss
        return {'l': loss, 'lr': self.scheduler.get_last_lr()[0]}, loss
    def validate_step(self, sample):
        tensor_hr = sample['high_res_tensor']
        tensor_lr = sample['low_res_tensor']
        mask = sample['mask']
        
        predictions = self.model(tensor_lr)
        if predictions.shape[2:] != tensor_hr.shape[2:]: #debugging/error circumventing - this line shouldn't run
            predictions = F.interpolate(predictions, size=313, mode='bilinear', align_corners=False)
            #if you are here, then that means sizing has been screwed up above
        loss = hybrid_ssim_mse(predictions * mask, tensor_hr * mask)
        return loss.item()

#outdated
    '''
    def train(self, max_epochs): #primary training loop
        train_losses=[]
        for epoch in range(max_epochs):
            print(f"Epoch {epoch+1}/{max_epochs}")
            total_train_loss = 0.0
            for batch in tqdm(self.train_loader, desc="Training"):
                batch = self.move_to_device(batch)
                loss = self.training_step(batch)
                total_train_loss += loss
            print(f"Epoch {epoch+1} Loss: {total_train_loss:.4f}")
            self.scheduler.step()
            #validate your hard work!
            self.validate(epoch)
            train_losses.append(total_train_loss/)
            checkpoint_path = 'checkpoints/RRDBENC_checkpoint.pth' #save after every epoch
            save_checkpoint(epoch, self.model, self.optimizer, train_losses, val_losses, checkpoint_path)
    '''
    def validate(self, epoch):
        #Validation loop to compute average loss over all samples in the validation set.
        #Returns float - Average validation loss for the epoch.
        self.model.eval()
        total_val_loss = 0.0
        num_samples = 0
    
        with torch.no_grad():  # Ensure no gradients are computed
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = self.move_to_device(batch)  # Move batch to GPU
    
                # Process each sample in the batch
                for i in range(hparams['eval_batch_size']):  # Iterate over batch size
                    sample = {
                        'high_res_tensor': batch['high_res_tensor'][i],
                        'low_res_tensor': batch['low_res_tensor'][i],
                        'mask': batch['mask'][i]
                    }
                    loss = self.validate_step(sample)
                    total_val_loss += loss
                    num_samples += 1
    
        avg_val_loss = total_val_loss / num_samples
        print(f"Validation Loss at Epoch {epoch + 1}: {avg_val_loss:.4f}")
        return avg_val_loss

    def test(self):
        """
        Testing loop to compute average loss over all samples in the test set.
    
        Returns:
            float: Average test loss.
        """
        self.model.eval()
        total_test_loss = 0.0
        num_samples = 0
    
        with torch.no_grad():  # Ensure no gradients are computed
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = self.move_to_device(batch)  # Move batch to GPU
    
                # Process each sample in the batch
                for i in range(hparams['eval_batch_size']):  
                    sample = {
                        'high_res_tensor': batch['high_res_tensor'][i],
                        'low_res_tensor': batch['low_res_tensor'][i],
                        'mask': batch['mask'][i:]
                    }
                    loss = self.test_step(sample)
                    total_test_loss += loss
                    num_samples += 1
    
        avg_test_loss = total_test_loss / num_samples
        print(f"Test Loss: {avg_test_loss:.4f}")
        return avg_test_loss
    def test_step(self, sample):
        return self.validate_step(sample)  # Same logic as validation step

def train_rrdb_spatial(model_task, max_epochs):
    #Train the RRDB-based model - this is what will help perform feature extraction
    # note model task is an object encapsulating the model, optimizer, scheduler, and data loaders.
    print("Starting training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_task.model.to(device)
    print(f"Model moved to {device}.")

    optimizer = model_task.optimizer
    scheduler = model_task.scheduler
    print("Optimizer and scheduler initialized.")
    training_losses = []
    validation_losses = []
    save_path = "model_rrdb_spatial_checkpoint.pth"  # Update path as needed
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1} of {max_epochs}")
        total_train_loss = 0.0
        for batch in tqdm(model_task.train_loader, desc="Training"):
            batch = model_task.move_to_device(batch)  # Move batch to GPU
            optimizer.zero_grad()
            num_samples=0
            batch_loss=0
            for i in range(hparams['batch_size']):  #hardcoding sorry
                sample = {
                    'high_res_tensor': batch['high_res_tensor'][i],
                    'low_res_tensor': batch['low_res_tensor'][i],
                    'mask': batch['mask'][i]
                }
                _, loss = model_task.training_step(sample)
                batch_loss += loss  # Accumulate batch loss

            # Average batch loss and backpropagate
            batch_loss = batch_loss / batch['high_res_tensor'].size(0)
            batch_loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            total_train_loss += batch_loss.item()  # Add batch loss to epoch loss

        # Compute average loss for the epoch
        avg_train_loss = total_train_loss / len(model_task.train_loader)
        training_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} Training Loss: {training_losses[-1]:.4f}")

        print("Validating...")
        validation_loss = model_task.validate(epoch)
        validation_losses.append(validation_loss)
        scheduler.step()

    #save all the model/training details
    print("Saving model...")
    torch.save({
        'epoch': max_epochs,
        'model_state_dict': model_task.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'training_losses': training_losses,
        'validation_losses': validation_losses,
    }, save_path)
    print(f"Model saved to {save_path}")

def hybrid_ssim_mse(predictions, targets, alpha=0.2, beta=0.8, weight_factor=30):
    
    #WE want to weight the spatial coordinates more in loss calculations as well as incorporate SSIM
    #SSIM should have a lower weight as we have effectively created "soft bounds" by utilizing a spatially encoded 
    #grid with coordinates embedded as features
    #predictions and targets must be on CPU and detached for NumPy conversion!!
    #ssim module we are using requires it all to be numpy
    predictions_np = predictions.squeeze(0).detach().cpu().numpy()
    targets_np = targets.squeeze(0).detach().cpu().numpy()

    #IMPORTANT NOTE: multichannel SSMI expects (H, W, C) format!!! 
    print(predictions_np.shape,targets_np.shape) #debugging
    predictions_np = predictions_np.transpose(1, 2, 0)  #hwc
    targets_np = targets_np.transpose(1, 2, 0)          #hwc format

    #multichannel since we have multiple channels/features (genes/spatial coords) - I wish I knew of this before
    ssim_score = ssim(
        predictions_np,
        targets_np,
        data_range=targets_np.max() - targets_np.min(),
        multichannel=True
    )
    print("ssim: ",ssim_score) 
    #we want to add some weight to the penalty for misrepresenting spatial coordinates due to their scale
    mse = (predictions - targets) ** 2
    weights = torch.ones_like(predictions)
    weights[:, -2:, :, :] *= weight_factor  #last two channels are x and y
    weighted_mse = torch.mean(mse * weights)
    hybrid_loss = alpha * (1 - ssim_score) + beta * weighted_mse #80% MSE and 20% SSM since soft bounds are effectively in existence within our grid
    return hybrid_loss