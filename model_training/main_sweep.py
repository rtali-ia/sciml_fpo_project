import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import argparse
import os
import torch
import gc
import wandb
from torch.utils import data
import random
import numpy as np
import time
from data.dataset_fpo import FPODataset
from models.deeponet.deeponet import DeepONet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    

def time_sampler(t_start, t_end, n_in, n_out): #Ronak - Added this
    """
    Generates time samples for training and testing.
    
    Args:
        t_start (int): The starting time index.
        t_end (int): The ending time index.
        n_in (int): The number of input time steps.
        n_out (int): The number of output time steps.
    
    Returns:
        tuple: A tuple containing the input and output time indices.
    """
    time_in = np.arange(t_start, t_end - n_out)
    time_out = np.arange(t_start + n_in, t_end)
    
    return time_in, time_out
    
def main(model_name, equation, config=None):
    
    torch.set_float32_matmul_precision('high')
    
    if config is None:
        # Load the configuration file
        config_path = f'configs/{equation}/{model_name}/conf.yaml'
        config = OmegaConf.load(config_path)
    else:
        # Merge the Wandb config with the base config
        config_path = f'configs/{equation}/{model_name}/conf.yaml'
        base_config = OmegaConf.load(config_path)
        # Ensure the Wandb config is nested correctly under 'model'
        wandb_config = OmegaConf.create({'model': {k: v for k, v in config.items()}})
        config = OmegaConf.merge(base_config, wandb_config)

    seed = config.trainer.seed
    set_seed(seed)
    
    # Set the time indices for training and testing [Ronak - Added this]
    t_start = config.data.time_start
    t_end = config.data.time_end
    n_in = config.data.time_steps_in
    n_out = config.data.time_steps_out
    
    t_in, t_out = time_sampler(t_start, t_end, n_in, n_out)
    
    # Define parameters from the config [Ronak - Changed this]
    train_dataset = FPODataset(
        file_path_x=config.data.file_path_train_x,
        file_path_y=config.data.file_path_train_y,
        time_in=t_in,
        time_out=t_out,
        data_type=config.data.type, 
        equation=config.data.equation,
        inputs=config.data.inputs
        )
    
    test_dataset = FPODataset(
        file_path_x=config.data.file_path_test_x,
        file_path_y=config.data.file_path_test_y,
        time_in=t_in,
        time_out=t_out,
        data_type=config.data.type, 
        equation=config.data.equation,
        inputs=config.data.inputs
        )
    
    train_loader = data.DataLoader(train_dataset,
                                    batch_size=config.data.batch_size,
                                    shuffle=config.data.shuffle,
                                    drop_last=False,
                                    num_workers=6
                                    )
    test_loader = data.DataLoader(test_dataset,
                                    batch_size=config.data.batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=6
                                    )

    # Define required model parameters for each model
    model_params = {
        'fno': ['n_modes', 'in_channels', 'out_channels', 'hidden_channels', 'projection_channels', 'n_layers', 'lr', 'plot_path', 'log_file'],
        'cno': ['in_channels', 'out_channels', 'n_layers', 'in_size', 'out_size', 'N_res', 'lr', 'plot_path', 'log_file'],
        'uno': ['in_channels', 'out_channels', 'hidden_channels', 'uno_out_channels', 'n_layers', 'uno_n_modes', 'uno_scalings', 'lr', 'plot_path', 'log_file'],
        'unet': ['in_channels', 'out_channels', 'enc_chs', 'dec_chs', 'lr', 'plot_path', 'log_file'],
        'wno': ['in_channels', 'out_channels', 'shape', 'width', 'level', 'lr', 'plot_path', 'log_file'],
        'deeponet': ['input_channels_func', 'input_channels_loc', 'out_channels', 'branch_net_layers', 'trunk_net_layers', 'modes', 'lr', 'plot_path', 'log_file'],
        'pod-deeponet': ['input_channels_func', 'input_channels_loc', 'out_channels', 'branch_net_layers', 'y_mean_file', 'v_file', 'modes', 'lr', 'plot_path', 'log_file'],
        'geometric-deeponet': ['input_channels_func', 'input_channels_loc', 'out_channels', 'branch_net_layers', 'trunk_net_layers', 'modes', 'lr', 'plot_path', 'log_file'],
        'oformer': ['in_channels', 'out_channels', 'in_emb_dim', 'out_seq_emb_dim', 'heads', 'depth', 'res', 'latent_channels', 'lr', 'plot_path', 'log_file'],
        'lsm': ['in_channels', 'out_channels', 'width', 'num_token', 'num_basis', 'patch_size', 'padding', 'lr', 'plot_path', 'log_file'], 
        'gino': ['n_modes', 'in_channels', 'out_channels', 'hidden_channels', 'projection_channels', 'n_layers', 'lr', 'plot_path', 'log_file'],
        'gnot': ['trunk_size', 'branch_sizes', 'n_layers', 'n_hidden', 'n_head', 'n_inner', 'mlp_layers', 'out_channels', 'in_channels', 'ffn_dropout', 'attn_dropout', 
                 'horiz_fourier_dim', 'lr', 'plot_path', 'log_file'],
    }

    # Filter the parameters for the model
    if model_name not in model_params:
        raise ValueError(f"Unknown model name: {model_name}")

    params = {k: v for k, v in config.model.items() if k in model_params[model_name]}
    
    print(f"Model parameters for {model_name} have been successfully set.")

    # Initialize the model
    # if model_name == 'fno':
    #     model = FNO(**params)
    # elif model_name == 'cno':
    #     model = cno(**params)
    # elif model_name == 'uno':
    #     model = UNO(**params)
    # elif model_name == 'wno':
    #     model = WNO(**params)
    # elif model_name == 'deeponet':
    #     model = DeepONet(**params)
    # elif model_name == 'pod-deeponet':
    #     model = PODDeepONet(**params)
    # elif model_name == 'geometric-deeponet':
    #     model = GeometricDeepONet(**params)
    if model_name == 'deeponet':
        model = DeepONet(**params)        
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Define the callbacks
    wandb_logger = WandbLogger(
        project=model_name, 
        config=config, 
        log_model='false'
        )
    
    
# Log the conf.yaml file
    wandb_logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))
    wandb_logger.experiment.save(config_path, policy="now")

    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.checkpoint.monitor,
        dirpath=os.path.join(wandb_logger.experiment.dir, f'{model_name}_checkpoints'),
        filename=config.callbacks.checkpoint.filename,
        save_top_k=config.callbacks.checkpoint.save_top_k,
        mode=config.callbacks.checkpoint.mode,
        every_n_epochs=100
    )

    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=wandb_logger
    )


    # Measure the time taken for training and testing
    start_time = time.time()
    
    # Train the model
    trainer.fit(model, train_loader, test_loader)

    # Test the model
    #trainer.test(model, test_loader)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

    # Log the elapsed time to Wandb
    wandb_logger.log_metrics({'training_time': elapsed_time})
    
    # Clean up to free GPU memory
    del model, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()


def sweep(model_name, equation):
    # Load the configuration file to get sweep parameters
    config_path = f'configs/{equation}/{model_name}/conf.yaml'
    base_config = OmegaConf.load(config_path)
    sweep_parameters = base_config.sweep_parameters

    # Convert the sweep parameters to a dictionary
    sweep_parameters = OmegaConf.to_container(sweep_parameters, resolve=True)

    # Define sweep configuration
    sweep_configuration = {
        'method': 'grid',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'val_loss_full'},
        'parameters': {k: v for k, v in sweep_parameters.items()}
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=model_name)

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            # Nest Wandb config correctly under 'model'
            wandb_config = OmegaConf.create({'model': {k: v for k, v in config.items()}})
            main(model_name, equation, config=wandb_config)

    # Start the sweep
    wandb.agent(sweep_id, function=train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with a specified configuration.")
    parser.add_argument('--model', type=str, required=True, help='Name of the model to train.')
    parser.add_argument('--equation', type=str, required=True, help='equation name for config dir')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep.')
    args = parser.parse_args()

    if args.sweep:
        sweep(args.model, args.equation)
    else:
        main(args.model, args.equation)
