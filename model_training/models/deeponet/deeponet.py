import torch.nn as nn
from models.base import BaseLightningModule
from models.deeponet.network import DeepONet2D
import torch.nn.functional as F
from sklearn.metrics import r2_score
import os
import sys
import numpy as np
import torch

class DeepONet(BaseLightningModule):
    def __init__(self, input_channels_func, input_channels_loc, out_channels, branch_net_layers, trunk_net_layers, modes, loss=nn.MSELoss(), lr=1e-4, plot_path='./plots/', log_file='DeepONet_log.txt',  epoch_per_timestep = 100, delta_time_step = 1):
        super(DeepONet, self).__init__(lr=lr, plot_path=plot_path, log_file=log_file, epoch_per_timestep = epoch_per_timestep, delta_time_step = delta_time_step)        

        self.input_channels_func = input_channels_func
        self.input_channels_loc = input_channels_loc
        self.output_channels = out_channels
        self.branch_net_layers = branch_net_layers
        self.trunk_net_layers = trunk_net_layers
        self.modes = modes
        self.loss = loss

        self.model = DeepONet2D(input_channels_func = self.input_channels_func, input_channels_loc = self.input_channels_loc, 
                                output_channels = self.output_channels, branch_net_layers = self.branch_net_layers,
                                trunk_net_layers = self.trunk_net_layers, modes = self.modes)
        
    def forward(self, x):
        x1 = x[:,:self.input_channels_func,:,:] # x[:,:,:,:3]# function values [Re, SDF, mask]
        x2 = x[:,self.input_channels_func:,:,:] #x[:,:,:,3:] # grid points [x,y]
        return self.model(x1, x2)


    def on_train_epoch_end(self): # Ronak - Added this. Will move it later to base.py
        #Method called at the end of the training epoch. We want to save the model prediction and update the time indices for training data every 100 epochs.
        #Save the model prediction if the epoch is a multiple of 100
        if (self.current_epoch +1) % self.epoch_per_timestep == 0:
            if len(self.epoch_predictions) > 0:
                all_predictions = np.concatenate(self.epoch_predictions, axis=0)
                predictions_numpy = all_predictions

                self.epoch_predictions = []

                filename = f'previous_data_{self.run_id}.npy'
                save_path = os.path.join('../dataset_generation/runtime_prediction_data', filename)

                if not os.path.exists(save_path):
                    os.makedirs('../dataset_generation/runtime_prediction_data', exist_ok=True)#make this dynamic
                    np.save(save_path, predictions_numpy)
                else:
                    #If file exists, append the new data to the second dimension of the existing data.
                    previous_data = np.load(save_path)
                    new_data = predictions_numpy
                    new_data = np.concatenate((previous_data, new_data), axis=1)
                    
                    #Save the updated data
                    np.save(save_path, new_data)
            
        
            #Call the update_data method of the datamodule to update the time indices for training data.
            self.trainer.datamodule.update_data(self.trainer.current_epoch, update_type = 'pred', file_path_xprime=save_path, epoch_per_timestep = self.epoch_per_timestep, delta_time_step = self.delta_time_step)