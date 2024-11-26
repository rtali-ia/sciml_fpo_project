import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scOT.model import ScOT, ScOTConfig




class scOTTimeDependent(pl.LightningModule):
    def __init__(self, image_size, in_channels, out_channels, depths, embed_dim, pretrained_path=None, loss=nn.MSELoss(), lr=1e-3, plot_path='./plots/', log_file='fno_log.txt'):
        super(scOTTimeDependent, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = loss
        self.depths = depths
        self.embed_dim = embed_dim
        
        self.lr = lr
        self.plot_path = plot_path
        self.log_file = log_file
        
        
        self.model_config = ScOTConfig(
                                    image_size=self.image_size,
                                    patch_size=4,
                                    num_channels=in_channels,
                                    num_out_channels=out_channels,
                                    embed_dim=self.embed_dim,
                                    depths=self.depths,
                                    num_heads=[3, 6, 12, 24],
                                    skip_connections=[2, 2, 2, 0],
                                    window_size=16,
                                    mlp_ratio=4.0,
                                    qkv_bias=True,
                                    hidden_dropout_prob=0.0,  # default
                                    attention_probs_dropout_prob=0.0,  # default
                                    drop_path_rate=0.0,
                                    hidden_act="gelu",
                                    use_absolute_embeddings=False,
                                    initializer_range=0.02,
                                    layer_norm_eps=1e-5,
                                    p=1,
                                    channel_slice_list_normalized_loss=None,
                                    residual_model="convnext",
                                    use_conditioning=None,
                                    learn_residual=False,
                                )
        
        if pretrained_path is not None:
            self.model = ScOT.from_pretrained(pretrained_path, config=self.model_config, ignore_mismatched_sizes=True)
        else:
            self.model = ScOT(config=self.model_config)

    def forward(self, x, t):
        return self.model(
            pixel_values=x,
            time=t
            )["output"] 
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch):
        x, y, t, _ = batch
        y_hat = self(x, t)
        
        # apply boundary conditions if data is FPO
        # for now, we'll determine this by number of input channels in x
        # if x.shape[1] == 6:
            
        
        loss = self.loss(y_hat[:,:,:,:128], y[:,:,:,:128])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        x, y, t, _ = batch
        y_hat = self(x, t)
        losses = self.custom_loss(y_hat, y)
        return losses[0]  # Returning the full MSE loss for monitoring
    
    def custom_loss(self, y_hat, y):
        loss = F.mse_loss(y_hat[:,:,:,:128], y[:,:,:,:128], reduction='none')
        losses = [(loss.sum((1,2,3))).mean().item()]
        for idx in range(y_hat.shape[1]):
            losses.append((loss[:,idx].sum((1,2))).mean().item())
        
        if y_hat.shape[1] == 4:
            loss_names = ['full', 'c', 'u', 'v', 'p']
        elif y_hat.shape[1] == 3:
            loss_names = ['full', 'u', 'v', 'p']
        for i, name in enumerate(loss_names):
            self.log(f'val_loss_{name}', losses[i], on_epoch=True, sync_dist=True)
        return losses