import gc
import os
import time
import wandb
import torch
import argparse
import random
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from data.dataset_fpo import FPODataModule
from models.deeponet.deeponet import DeepONet
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

def main(model_name, equation, config=None):
    torch.set_float32_matmul_precision('high')

    config = {
        "data": {
            "file_path_train_x": "/scratch/au2216/flowbench_fpo/sciml_fpo_project/model_training/X_train_new_batch.npz",
            "file_path_train_y": "/scratch/au2216/flowbench_fpo/sciml_fpo_project/model_training/Y_train_new_batch.npz",
            "tmax": 5,
            "in_start": 0,
            "out_start": 1,
            "batch_size": 4
        },
        "trainer": {
            "seed": 0,
            "epoch_per_timestep":100,
            "delta_time_step": 1,
            "accelerator": 'gpu',
            "devices": 1,
            "log_every_n_steps": 10
        },
        "model": {
            "steps_in": 3,
            "steps_out": 1,
            "branch_net_layers": [512, 512, 512],
            "trunk_net_layers": [256, 256, 256],
            "modes": 128,
            "input_channels_loc": 2
        },
        "callbacks": {
            "checkpoint": 'checkpoints',
            "filename": 'fpo-{epoch:02d}-{val_loss:.4f}',
            "monitor": 'val_r2_score_full',
            "mode": 'min',
            "save_top_k": 3,
            "every_n_epochs": 50,
            "save_last": True
        }
    }

    # Convert dictionary to OmegaConf
    config = OmegaConf.create(config)
    
    seed = config.trainer.seed
    set_seed(seed)

    # Create a serializable config for wandb
    config_serializable = OmegaConf.to_container(config, resolve=True)
    
    wandb_logger = WandbLogger(
        project=model_name, 
        config=config_serializable, 
        log_model=False  # Changed 'false' string to False boolean
    )

    wandb_logger.experiment.config.update(config_serializable)

    data_module = FPODataModule(
        file_path_X_train=config.data.file_path_train_x,
        file_path_Y_train=config.data.file_path_train_y,
        file_path_X_val=config.data.file_path_train_x,
        file_path_Y_val=config.data.file_path_train_y,
        tmax=config.data.tmax,
        steps_in=config.model.steps_in,
        steps_out=config.model.steps_out,
        in_start=config.data.in_start,
        out_start=config.data.out_start,
        batch_size=config.data.batch_size
    )

    model = DeepONet(
        input_channels_func=(config.model.steps_in + 1) * 3,
        input_channels_loc=config.model.input_channels_loc,
        out_channels=config.model.steps_out * 3,
        branch_net_layers=config.model.branch_net_layers,
        trunk_net_layers=config.model.trunk_net_layers,
        modes=config.model.modes,
        epoch_per_timestep = config.trainer.epoch_per_timestep, 
        delta_time_step = config.trainer.delta_time_step
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(wandb_logger.experiment.dir, f'{model_name}_checkpoints'),
        filename=config.callbacks.filename,
        monitor=config.callbacks.monitor,
        mode=config.callbacks.mode,
        save_top_k=config.callbacks.save_top_k,
        every_n_epochs=config.callbacks.every_n_epochs,
        save_last=config.callbacks.save_last
    )

    max_steps = ((config["data"]["tmax"] - max(config["data"]["in_start"] + config["model"]["steps_in"], config["data"]["out_start"] + config["model"]["steps_out"])) // config["trainer"]["delta_time_step"]) + 1

    trainer = pl.Trainer(
        max_epochs=config.trainer.epoch_per_timestep*max_steps,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        callbacks=[checkpoint_callback],
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_progress_bar=True,
        logger=wandb_logger
    )

    start_time = time.time()
    trainer.fit(model, datamodule=data_module)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")
    wandb_logger.log_metrics({'training_time': elapsed_time})

    # Cleanup
    del model, data_module
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()

def sweep(model_name, equation):
    pass

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
