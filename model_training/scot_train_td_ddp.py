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
from data.time_dependent_dataset import MultiPhysicsBubbleDataset, FlowPastObjectDataset
from models.scot.scOT_timedependent import scOTTimeDependent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


def main(model_name, equation, config=None):
    torch.set_float32_matmul_precision('high')

    if config is None:
        # Load the configuration file
        config_path = f'configs/neurips_fm_wrkshp/{equation}/{model_name}/conf.yaml'
        config = OmegaConf.load(config_path)
    else:
        # Merge the Wandb config with the base config
        config_path = f'configs/neurips_fm_wrkshp/{equation}/{model_name}/conf.yaml'
        base_config = OmegaConf.load(config_path)
        wandb_config = OmegaConf.create({'model': {k: v for k, v in config.items()}})
        config = OmegaConf.merge(base_config, wandb_config)

    seed = config.trainer.seed
    set_seed(seed)

    # Define datasets based on equation
    if equation == 'multiphysics_bubble':
        train_dataset = MultiPhysicsBubbleDataset(
            file_path=config.data.file_path,
            train=True,
            partial_data=True
        )
        test_dataset = MultiPhysicsBubbleDataset(
            file_path=config.data.file_path,
            train=False
        )
    elif equation == 'fpo':
        train_dataset = FlowPastObjectDataset(
            file_path=config.data.file_path,
            train=True
        )
        test_dataset = FlowPastObjectDataset(
            file_path=config.data.file_path,
            train=False
        )

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=config.data.batch_size,
                                   shuffle=True,
                                   drop_last=False,
                                   num_workers=6)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=config.data.batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=6)

    # Define model parameters based on the model name
    model_params = {
        'scot-T': ['image_size', 'in_channels', 'out_channels', 'depths', 'embed_dim', 'pretrained_path', 'lr', 'plot_path', 'log_file'],
        'poseidon-T': ['image_size', 'in_channels', 'out_channels', 'depths', 'embed_dim', 'pretrained_path', 'lr', 'plot_path', 'log_file'],
        'scot-B': ['image_size', 'in_channels', 'out_channels', 'depths', 'embed_dim', 'pretrained_path', 'lr', 'plot_path', 'log_file'],
        'poseidon-B': ['image_size', 'in_channels', 'out_channels', 'depths', 'embed_dim', 'pretrained_path', 'lr', 'plot_path', 'log_file'],
        'scot-L': ['image_size', 'in_channels', 'out_channels', 'depths', 'embed_dim', 'pretrained_path', 'lr', 'plot_path', 'log_file'],
        'poseidon-L': ['image_size', 'in_channels', 'out_channels', 'depths', 'embed_dim', 'pretrained_path', 'lr', 'plot_path', 'log_file']
    }

    if model_name not in model_params:
        raise ValueError(f"Unknown model name: {model_name}")

    params = {k: v for k, v in config.model.items() if k in model_params[model_name]}

    # Initialize the model
    model = scOTTimeDependent(**params)

    # Initialize the WandbLogger only on the main process (rank 0)
    if int(os.getenv('LOCAL_RANK', 0)) == 0:
        wandb_logger = WandbLogger(
            project=model_name,
            config=config,
            log_model=False
        )
        # Log hyperparameters to Wandb
        wandb_logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
    else:
        wandb_logger = None

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.checkpoint.monitor,
        dirpath=os.path.join(wandb_logger.experiment.dir if wandb_logger else '', f'{model_name}_checkpoints'),
        filename=config.callbacks.checkpoint.filename,
        save_top_k=config.callbacks.checkpoint.save_top_k,
        mode=config.callbacks.checkpoint.mode,
        every_n_epochs=config.callbacks.checkpoint.every_n_epochs,
    )

    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        strategy='ddp',
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=wandb_logger if wandb_logger else False
    )

    # Measure training time
    start_time = time.time()

    # Train the model
    trainer.fit(model, train_loader, test_loader)

    # Training time calculation
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

    # Log the elapsed time to Wandb (only if wandb_logger is initialized)
    if wandb_logger:
        wandb_logger.log_metrics({'training_time': elapsed_time})

    # Clean up GPU memory
    del model, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    if wandb_logger:
        wandb.finish()


def sweep(model_name, equation):
    # Load the configuration file to get sweep parameters
    config_path = f'configs/neurips_fm_wrkshp/{equation}/{model_name}/conf.yaml'
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
