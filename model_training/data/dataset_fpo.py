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
        self.collocation = False
        
        # Load data from .npz files
        x_indices = [0,1,2] + [3 + t*3 + i for t in time_in for i in range(3)] # shape [num_samples, [Re, SDF, Mask, u1, v1, p1, u2, v2, ...], 256, 1024]
        y_indices = [t*3 + i for t in time_out for i in range(3)] # shape [num_samples, [u_k, v_k, p_k, u_k+1, v_k+1, p_k+1], 256, 1024]

        #Ronak - Removing ['data] as directly providing npy files
        with np.load(file_path_x, mmap_mode='r') as data:
            x = data['arr_0'][:,x_indices,:,:].copy()

        with np.load(file_path_y, mmap_mode='r') as data:
            y = data['arr_0'][:,y_indices,:,:].copy()
  
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
    def __init__(self, file_path_x, file_path_xprime, file_path_y, time_in_gt, time_out_gt, equation=None, data_type=None, inputs=None, geometric_deeponet=False):
        """
        Initializes the dataset with the paths to the .npz files and an optional transform.
        
        Args:
            file_path_x (str): Path to the .npz file containing the input data.
            file_path_y (str): Path to the .npz file containing the target data.
        """
        
        # Load data from .npz files
        # Load data from .npz files
        x_indices = [0,1,2] + [3 + t*3 + i for t in time_in_gt for i in range(3)] # shape [num_samples, [Re, SDF, Mask, u1, v1, p1, u2, v2, ...], 256, 1024]
        y_indices = [t*3 + i for t in time_out_gt for i in range(3)] # shape [num_samples, [u_k, v_k, p_k, u_k+1, v_k+1, p_k+1], 256, 1024]

        #Ronak - Removing ['data] as directly providing npy files
        with np.load(file_path_x, mmap_mode='r') as data:
            x = data['arr_0'][:,x_indices,:,:].copy()

        with np.load(file_path_y, mmap_mode='r') as data:
            y = data['arr_0'][:,y_indices,:,:].copy()

        xprime = np.load(file_path_xprime)
        
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
    def __init__(self, file_path_X_train, file_path_Y_train, file_path_X_val, file_path_Y_val , tmax, steps_in, steps_out, in_start, out_start, batch_size=4, shuffle=False, num_workers=4):
        """
        Initializes the data module with the training and validation data.
        tmin and tmax are the time indices for the training and validation data loaded from the .npz files.
        steps_in and steps_out are the number of input and output time steps that we want to use for training.
        The idea is to have a sliding window of time steps for training and testing.
        
        Args:
        
        tmax (int): The maximum time index in the preprocessed dataset.
        steps_in (int): The number of input time steps.
        steps_out (int): The number of output time steps that we want to predict. Defaults to 1 when we use reuse predictions for future predictions.
        in_start (int): The starting time index for the input data.
        out_start (int): The starting time index for the output data.
        
        """
        super().__init__()
        self.file_path_X_train = file_path_X_train
        self.file_path_Y_train = file_path_Y_train
        self.file_path_X_val = file_path_X_val
        self.file_path_Y_val = file_path_Y_val
        self.tmax = tmax
        self.steps_in = steps_in
        self.steps_out = steps_out
        self.in_start = in_start
        self.out_start = out_start

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.train_data_loader = FPODataset(self.file_path_X_train, self.file_path_Y_train, 
                                                list(range(self.in_start, self.in_start + self.steps_in)), 
                                                list(range(self.out_start, self.out_start + self.steps_out)), data_type='collocation'
                                                ) # Initialized with starting values but Will be updated in the update_data method
        self.val_data_loader = FPODataset(self.file_path_X_val, self.file_path_Y_val, 
                                                list(range(self.in_start, self.in_start + self.steps_in)), 
                                                list(range(self.out_start, self.out_start + self.steps_out)),data_type='collocation'
                                                ) # Initialized with starting values but Will be updated in the update_data method
        
    def train_dataloader(self):
        return DataLoader(
            self.train_data_loader,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data_loader,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
    
    def update_data(self, epoch, update_type='gt', file_path_xprime = None):
        
        """
        This contains the logic to create a shifting time window for training and validation data.
        
        Args:
            epoch (int): The current epoch number coming from on_epoch_end.
            update_type (str): The type of update to perform. 'gt' for ground truth update and 'pred' for prediction update.
            Ground truth update is used when we want to train the model on the actual data without reusing predictions
            Prediction update is used when we want to reuse the predictions from the previous time step to predict the next time step.
        
        """ 
        if update_type == 'gt':
            
            if epoch > 0 and epoch % 100 == 0:
                
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
                self.train_data_loader = FPODataset(self.file_path_X_train, self.file_path_Y_train, t_in, t_out, data_type='collocation')
                self.val_data_loader = FPODataset(self.file_path_X_val, self.file_path_Y_val, t_in, t_out, data_type='collocation')
                
        elif update_type == 'pred':
                
            if epoch > 0 and epoch % 100 == 0:
                if file_path_xprime == None:
                    raise ValueError("Invalid xprime path type. Please provide path for train and val dataset for xprime")
                
                offset = epoch//100
                
                # Update the time indices for training and validation data
                startin = self.in_start + offset # Move input window start point by 1 step
                startout = self.out_start + offset # Move output window start point by 1 step
                
                #Calculate the indices for the new time window
                t_in = np.arange(startin, startin + self.steps_in - offset) # input time array will be 1 step shorter. Reusing prediction is handled in the dataloader.
                t_out = np.arange(startout, startout + self.steps_out) # length of output time array will be same. BTW this will default to 1 for now.
                
                #Check if the new time window is within the bounds of the data
                if t_in[-1] > self.tmax or t_out[-1] > self.tmax:
                    raise ValueError("Time window exceeds data bounds. Please adjust the time windows.")
                
                #Update the training and validation datasets with the new time window
                self.train_data_loader = FPODatasetMix(self.file_path_X_train, file_path_xprime, self.file_path_Y_train, t_in, t_out, data_type='collocation')
                self.val_data_loader = FPODataset(self.file_path_X_val, self.file_path_Y_val, t_in, t_out, data_type='collocation')
        else:
            raise ValueError("Invalid update type. Please use 'gt'/'pred' for ground truth update.")