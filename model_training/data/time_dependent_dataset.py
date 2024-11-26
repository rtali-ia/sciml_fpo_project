import torch
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
import numpy as np
from itertools import product

class MultiPhysicsBubbleDataset(Dataset):
    '''
    Custom dataset for loading and processing multiPhysics bubble data from npz files.
    '''
    def __init__(self, file_path='/work/mech-ai/mehdish/MPF-bench/RawData/2D/bubble/dataset', train=True, partial_data=False):
        x_full = np.load(os.path.join(file_path, 'X_tensor.npz'), mmap_mode='r')['X']
        y_full = np.load(os.path.join(file_path, 'Y_tensor.npz'), mmap_mode='r')['Y']
        assert x_full.shape[0] == y_full.shape[0], 'Num trajectories in x and y files do not match'
        
        # Splitting train and test sets
        if train:
            if not partial_data:
                self.x = x_full[:int(x_full.shape[0] * 0.8)]
                self.y = y_full[:int(x_full.shape[0] * 0.8)]
            else:
                self.x = x_full[:int((x_full.shape[0] * 0.8) / 10.)]
                self.y = y_full[:int((x_full.shape[0] * 0.8) / 10.)]
        else:
            self.x = x_full[int(x_full.shape[0] * 0.8):]
            self.y = y_full[int(x_full.shape[0] * 0.8):]
            
        self.num_trajectories = self.x.shape[0]
        self.num_timesteps = self.y.shape[1]

        # Precompute all valid (input_idx, output_idx) pairs for every trajectory
        self.combinations = []
        for trajectory_idx in range(self.num_trajectories):
            valid_pairs = [(input_idx, output_idx) 
                           for input_idx in range(self.num_timesteps - 1) 
                           for output_idx in range(input_idx + 1, self.num_timesteps)]
            self.combinations.extend([(trajectory_idx, k, j) for (k, j) in valid_pairs])

    def __len__(self):
        # The total length is now the number of valid (k, j) pairs across all trajectories
        return len(self.combinations)

    def __getitem__(self, idx):
        # Retrieve the trajectory and timestep indices from the combinations list
        trajectory_idx, input_idx, output_idx = self.combinations[idx]
        info_dict = {'trajectory_idx': trajectory_idx,
                     'input_timestep': input_idx,
                     'output_timestep': output_idx
                     }
        
        # Sample the input (current timestep) and output (future timestep)
        xs = self.x[trajectory_idx, :, None, None] * np.ones((4, self.y.shape[-2], self.y.shape[-1]))
        input = np.concatenate((self.y[trajectory_idx, input_idx, :, :, :], xs), axis=0)
        output = self.y[trajectory_idx, output_idx, :, :, :]
        
        input_tensor = torch.tensor(input, dtype=torch.float32)
        output_tensor = torch.tensor(output, dtype=torch.float32)
        
        # pad to make square images (gross but required for scot model)
        padding = (0, 128)
        input_padded = F.pad(input_tensor, padding, "constant", 0)
        output_padded = F.pad(output_tensor, padding, "constant", 0)
        
        # Compute the lead time (normalized)
        lead_time = (output_idx - input_idx) / self.num_timesteps
        
        return input_padded, output_padded, torch.tensor(lead_time, dtype=torch.float32).unsqueeze(0), info_dict

        
        
class FlowPastObjectDataset(Dataset):
    '''
    Custom dataset for loading and processing flow past object data from npz files.
    '''
    def __init__(self, file_path='/work/mech-ai/edherron/data/flowbench/flow_past_object/skelneton/subsampled_data.npz', object_type='skelneton', train=True):
        data = np.load(file_path, mmap_mode='r')
        x_full = data['x']
        y_full = data['y']
        # x_full = np.load('/work/mech-ai/rtali/projects/all-neural-operators/TimeDependentNS/FPO_Data/512x128/skelneton_fpo_data_X.npz', mmap_mode='r')['data']
        # y_full = np.load('/work/mech-ai/rtali/projects/all-neural-operators/TimeDependentNS/FPO_Data/512x128/skelneton_fpo_data_Y.npz', mmap_mode='r')['data']

        assert x_full.shape[0] == y_full.shape[0], 'Num trajectories in x and y files do not match'
        
        # Splitting train and test sets
        if train:
            self.x = x_full[:int(x_full.shape[0] * 0.8)]
            self.y = y_full[:int(x_full.shape[0] * 0.8)]
        else:
            self.x = x_full[int(x_full.shape[0] * 0.8):]
            self.y = y_full[int(x_full.shape[0] * 0.8):]
            
        self.num_trajectories = self.x.shape[0]
        self.num_timesteps = self.y.shape[1]

        # Precompute all valid (input_idx, output_idx) pairs for every trajectory
        self.combinations = []
        for trajectory_idx in range(self.num_trajectories):
            valid_pairs = [(input_idx, output_idx) 
                           for input_idx in range(self.num_timesteps - 1) 
                           for output_idx in range(input_idx + 1, self.num_timesteps)]
            self.combinations.extend([(trajectory_idx, k, j) for (k, j) in valid_pairs])

    def __len__(self):
        # The total length is now the number of valid (k, j) pairs across all trajectories
        return len(self.combinations)

    def __getitem__(self, idx):
        # Retrieve the trajectory and timestep indices from the combinations list
        trajectory_idx, input_idx, output_idx = self.combinations[idx]
        info_dict = {'trajectory_idx': trajectory_idx,
                     'input_timestep': input_idx,
                     'output_timestep': output_idx
                     }
        
        # Sample the input (current timestep) and output (future timestep)
        input = np.concatenate((self.y[trajectory_idx, input_idx, :, :, :], self.x[trajectory_idx, :, :, :]), axis=0)
        output = self.y[trajectory_idx, output_idx, :, :, :]
        
        input_tensor = torch.tensor(input, dtype=torch.float32)
        output_tensor = torch.tensor(output, dtype=torch.float32)
        
        # pad to make square images (gross but required for scot model)
        padding = (0, 384)
        input_padded = F.pad(input_tensor, padding, "constant", 0)
        output_padded = F.pad(output_tensor, padding, "constant", 0)
        
        # Compute the lead time (normalized)
        lead_time = (output_idx - input_idx) / self.num_timesteps
        
        return input_padded, output_padded, torch.tensor(lead_time, dtype=torch.float32).unsqueeze(0), info_dict