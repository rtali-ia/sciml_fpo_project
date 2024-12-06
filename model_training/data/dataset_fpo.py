import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl

class FPODataset(Dataset):
    """
    Custom dataset for loading and processing Lid Driven Cavity problem data from .npz files.
    """
    def __init__(self, file_path_x, file_path_y, time_in, time_out, equation=None, data_type=None, inputs=None, geometric_deeponet=False):
        """
        Initializes the dataset with the paths to the .npz files and an optional transform.
        
        Args:
            file_path_x (str): Path to the .npz file containing the input data.
            file_path_y (str): Path to the .npz file containing the target data.
        """
        
        # Load data from .npz files
        #Ronak - Removing ['data] as directly providing npy files
        x = np.load(file_path_x)  # shape [num_samples, [Re, SDF, Mask, u1, v1, p1, u2, v2, ...], 256, 1024]
        y = np.load(file_path_y)  # shape [num_samples, [u_k, v_k, p_k, u_k+1, v_k+1, p_k+1], 256, 1024]
        
        #Select only the time_in and time_out data
        x = x[:,time_in,:,:]
        y = y[:,time_out,:,:]
  
        if data_type == 'field':
            self.collocation = False
        elif data_type == 'collocation':
            self.collocation = True
            self.resolution_x = x.shape[-1]
            self.resolution_y = x.shape[-2]

        # Convert numpy arrays to PyTorch tensors
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the sample and its label at the specified index.
        
        Args:Returns:
            tuple: (sample, target) where sample is the input data and target is the expected output.
        """
        sample = self.x[idx]
        target = self.y[idx]
        
        if self.collocation:
            grid = self.get_grid()
            sample = torch.cat((sample, grid), dim=0)
        
        return sample, target
    
    def get_grid(self):
        # Create the uniform grid for x and y locations of the input
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, self.resolution_x), 
            np.linspace(0, 1, self.resolution_y)
        )
        # Stack the grids along the last dimension to get (res, res, 2) shape
        grid = np.stack([grid_x, grid_y], axis=-1)
        # Transpose to get shape (2, res, res) instead of (res, res, 2)
        grid = grid.transpose(2, 0, 1)
        # Convert to PyTorch tensor
        grid = torch.tensor(grid, dtype=torch.float)
        return grid
    


class FPODatasetMix(Dataset):
    """
    Custom dataset for loading and processing Lid Driven Cavity problem data from .npz files.
    """
    def __init__(self, file_path_x, file_path_xprime, file_path_y, time_in_gt, time_out_gt, time_in_pred, equation=None, data_type=None, inputs=None, geometric_deeponet=False):
        """
        Initializes the dataset with the paths to the .npz files and an optional transform.
        
        Args:
            file_path_x (str): Path to the .npz file containing the input data.
            file_path_y (str): Path to the .npz file containing the target data.
        """
        
        # Load data from .npz files
        #Ronak - Removing ['data] as directly providing npy files
        x = np.load(file_path_x)  # shape [num_samples, [Re, SDF, Mask, u1, v1, p1, u2, v2, ...], 256, 1024]
        xprime = np.load(file_path_xprime) # shape [num_samples, [Re, SDF, Mask, u1, v1, p1, u2, v2, ...], 256, 1024]
        y = np.load(file_path_y)  # shape [num_samples, [u_k, v_k, p_k, u_k+1, v_k+1, p_k+1], 256, 1024]
        
        #Select only the time_in and time_out data
        x = x[:,time_in_gt,:,:]
        xprime = xprime[:,time_in_pred,:,:]
        y = y[:,time_out_gt,:,:]
        
        #Concatenate x and xprime along the channel dimension
        x = np.concatenate((x, xprime), axis=1)
        
        if data_type == 'field':
            self.collocation = False
        elif data_type == 'collocation':
            self.collocation = True
            self.resolution_x = x.shape[-1]
            self.resolution_y = x.shape[-2]

        # Convert numpy arrays to PyTorch tensors
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the sample and its label at the specified index.
        
        Args:Returns:
            tuple: (sample, target) where sample is the input data and target is the expected output.
        """
        sample = self.x[idx]
        target = self.y[idx]
        
        if self.collocation:
            grid = self.get_grid()
            sample = torch.cat((sample, grid), dim=0)
        
        return sample, target
    
    def get_grid(self):
        # Create the uniform grid for x and y locations of the input
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, self.resolution_x), 
            np.linspace(0, 1, self.resolution_y)
        )
        # Stack the grids along the last dimension to get (res, res, 2) shape
        grid = np.stack([grid_x, grid_y], axis=-1)
        # Transpose to get shape (2, res, res) instead of (res, res, 2)
        grid = grid.transpose(2, 0, 1)
        # Convert to PyTorch tensor
        grid = torch.tensor(grid, dtype=torch.float)
        return grid


class FPODataModule(pl.LightningDataModule):
    def __init__(self, X_train, Y_train, X_val, Y_val, tmin, tmax, steps_in, steps_out, in_start, out_start):
        """
        Initializes the data module with the training and validation data.
        tmin and tmax are the time indices for the training and validation data loaded from the .npz files.
        steps_in and steps_out are the number of input and output time steps that we want to use for training.
        The idea is to have a sliding window of time steps for training and testing.
        """
        super().__init__()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.tmin = tmin
        self.tmax = tmax
        self.steps_in = steps_in
        self.steps_out = steps_out
        self.in_start = in_start
        self.out_start = out_start
        self.train_dataset = None
        self.val_dataset = None
        
        def train_dataloader(self):
            return self.train_dataset
        
        def val_dataloader(self):
            return self.val_dataset
        
        def save_predicted_data(self, predicted_data, save_path = './dataset_generation/runtime_prediction_data/previous_data.npy'):
            """
            Saves the predicted data to a .npy file.
            """
            np.save(save_path, predicted_data)
            
        
        def update_data(self, epoch, update_type='gt'):
            
            """
            This contains the logic to create a shifting time window for training and validation data.
            """
            if update_type == 'gt':
                if epoch%100 == 0:
                    
                    offset = epoch//100
                    
                    # Update the time indices for training and validation data
                    startin = self.in_start + offset
                    startout = self.out_start + offset
                    
                    #Calculate the indices for the new time window
                    t_in = np.arange(startin, startin + self.steps_in )
                    t_out = np.arange(startout, startout + self.steps_out)
                    
                    #Check if the new time window is within the bounds of the data
                    if t_in[-1] > self.tmax or t_out[-1] > self.tmax:
                        raise ValueError("Time window exceeds data bounds. Please adjust the time windows.")
                    
                    #Update the training and validation datasets with the new time window
                    self.train_dataset = FPODataset(self.X_train, self.Y_train, self.tmin, self.tmax, t_in, t_out)
                    self.val_dataset = FPODataset(self.X_val, self.Y_val, self.tmin, self.tmax, t_in, t_out)
                    
            elif update_type == 'pred':
                if epoch % 100 == 0:
                    
                    offset = epoch//100
                    
                    # Update the time indices for training and validation data
                    startin = self.in_start + offset
                    startout = self.out_start + offset
                    
                    #Calculate the indices for the new time window
                    t_in = np.arange(startin, startin + self.steps_in - offset)
                    t_out = np.arange(startout, startout + self.steps_out)
                    
                    #Check if the new time window is within the bounds of the data
                    if t_in[-1] > self.tmax or t_out[-1] > self.tmax:
                        raise ValueError("Time window exceeds data bounds. Please adjust the time windows.")
                    
                    #Update the training and validation datasets with the new time window
                    self.train_dataset = FPODataset(self.X_train, self.Y_train, self.tmin, self.tmax, t_in, t_out)
                    self.val_dataset = FPODataset(self.X_val, self.Y_val, self.tmin, self.tmax, t_in, t_out)
            else:
                raise ValueError("Invalid update type. Please use 'gt'/'pred' for ground truth update.")
                
                
        
        
        