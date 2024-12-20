import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score


class BaseLightningModule(pl.LightningModule):
    def __init__(self, lr, plot_path, log_file, **kwargs):
        super(BaseLightningModule, self).__init__()
        self.lr = lr
        self.plot_path = plot_path
        self.log_file = log_file

        # Add storage for batch data
        self.epoch_predictions = []

    def forward(self):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch):
        self.current_batch = batch

        x, y = batch
        # apply object boundary to prediction before computing loss
        y_hat = self(x)
        for i in range(y_hat.shape[1]):
            y_hat[:,i,:,:] = torch.where(x[:,-1,:,:] > 0.,
                                         y_hat[:,i,:,:],
                                         torch.zeros_like(y_hat[:,i,:,:].type_as(y_hat))
                                         )

        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)

        # Store predictions during training
        if self.current_epoch > 0 and self.current_epoch % 100 == 0:
            self.epoch_predictions.append(y_hat.cpu().detach())

        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        losses = self.custom_loss(y_hat, y, x)
        
        '''
        Bring to CPU and convert to numpy for r2 score calculation
        '''
        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        
        # Take the mean value of the last two dimensions (x and y)
        y_mean = y[:,:,4,4]
        y_hat_mean = y_hat[:,:,4,4]
        
        # Calculate r2 score
        for i in range(y_hat.shape[1]):
            r2_channel = r2_score(y_mean[:, i].flatten(), y_hat_mean[:, i].flatten())
            self.log(f'val_r2_score_channel_{i}', r2_channel, on_epoch=True)
        
        #Log total r2 score
        r2_overall = r2_score(y_mean.flatten(), y_hat_mean.flatten())
        self.log('val_r2_score_full', r2_overall, on_epoch=True)
       
        return losses[0]  # Returning the full MSE loss for monitoring
    
    def custom_loss(self, y_hat, y, x):
        # Apply BCs, find area of domain outside of object, collect losses
        # for mask input type this will be an identity op
        node_count = torch.where(x[:,-1,:,:] > 0., 
                          torch.ones_like(x[:,-1,:,:]).type_as(x), 
                          torch.zeros_like(x[:,-1,:,:]).type_as(x)
                          ).sum((1,2))
        
        y_hat = self(x)
        for i in range(y_hat.shape[1]):
            y_hat[:,i,:,:] = torch.where(x[:,-1,:,:] > 0.,
                                         y_hat[:,i,:,:],
                                         torch.zeros_like(y_hat[:,i,:,:].type_as(y_hat))
                                         )
            
        loss = F.mse_loss(y_hat, y, reduction='none')
        
        losses = [(loss.sum((1,2,3)) / node_count).mean().item()]
        
        for idx in range(y_hat.shape[1]):
            losses.append((loss[:,idx].sum((1,2)) / node_count).mean().item())
        
        if y_hat.shape[1] == 1:    # c for ns only
            loss_names = ['full','c']
        if y_hat.shape[1] == 2:    # uv for ns only
            loss_names = ['full','u', 'v']
        if y_hat.shape[1] == 3:    # uvp for ns and fpo only
            loss_names = ['full','u', 'v', 'p']
        if y_hat.shape[1] == 4:    # uvpt for ns+ht
            loss_names = ['full','u', 'v', 'p', 't']
        if y_hat.shape[1] == 8:    # uvptc for ns+ht
            loss_names = ['full','x0', 'y0', 'z0', 'theta', 'phi', 'a', 'b','c']
        else:    # uvpt for ns+ht
            # Generate loss names with numbered components
            # Abhisek - better way of handing fpo need
            loss_names = ['full']
            num_sets = y_hat.shape[1]//3  # For components 1-10
            components = ['u', 'v', 'p']
            for i in range(1, num_sets + 1):
                for comp in components:
                    loss_names.append(f'{comp}{i}')
          
        return losses
