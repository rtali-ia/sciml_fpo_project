import torch.nn as nn
from models.base import BaseLightningModule
from models.deeponet.network import DeepONet2D
import torch.nn.functional as F
from sklearn.metrics import r2_score

class DeepONet(BaseLightningModule):
    def __init__(self, input_channels_func, input_channels_loc, out_channels, branch_net_layers, trunk_net_layers, modes, loss=nn.MSELoss(), lr=1e-4, plot_path='./plots/', log_file='DeepONet_log.txt'):
        super(DeepONet, self).__init__(lr=lr, plot_path=plot_path, log_file=log_file)
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

    def on_train_epoch_end(self, x): # Ronak - Added this. Will move it later to base.py
        """
        Method called at the end of the training epoch. We want to save the model prediction and update the time indices for training data every 100 epochs.
        """
        
        #Save the model prediction if the epoch is a multiple of 100
        if self.current_epoch % 100 == 0:
            
            #Reuse the input data to save the model prediction.
            x_store = x[:,:self.input_channels_func,:,:] #Same as x1 in the forward method.
            
            #Get the model prediction
            y_hat = self.model(x_store)
    
            #Save the model prediction
            self.trainer.datamodule.save_predicted_data(y_hat, save_path = './dataset_generation/runtime_prediction_data/previous_data.npy')
        
            #Call the update_data method of the datamodule to update the time indices for training data.
            self.trainer.datamodule.update_data(self.trainer.current_epoch)
    