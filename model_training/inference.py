import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from data.dataset import LidDrivenDataset
from data.graph_dataset import LidDrivenGraphDataset
from models.fno.fno import FNO
from models.cno.cno import cno
from models.uno.uno import UNO
from models.unet.unet import UNetTrainer
from models.wno.wno import WNO
from models.deeponet.deeponet import DeepONet
from models.pod_deeponet.pod_deeponet import PODDeepONet
from models.geometric_deeponet.geometric_deeponet import GeometricDeepONet
from models.oformer.oformer import OFormer
from models.gnot.gnot import GNOT
from models.lsm.lsm import LSMTrainer

def load_model(model_name, config):
    model_params = {
        'fno': ['n_modes', 'in_channels', 'out_channels', 'hidden_channels', 'projection_channels', 'n_layers', 'lr', 'plot_path', 'log_file'],
        'cno': ['in_channels', 'out_channels', 'n_layers', 'in_size', 'out_size', 'N_res', 'lr', 'plot_path', 'log_file'],
        'uno': ['in_channels', 'out_channels', 'hidden_channels', 'uno_out_channels', 'n_layers', 'uno_n_modes', 'uno_scalings', 'lr', 'plot_path', 'log_file'],
        'unet': ['in_channels', 'out_channels', 'enc_chs', 'dec_chs', 'lr', 'plot_path', 'log_file'],
        'wno': ['in_channels', 'out_channels', 'shape', 'width', 'level', 'lr', 'plot_path', 'log_file'],
        'deeponet': ['input_channels_func', 'input_channels_loc', 'output_channels', 'branch_net_layers', 'trunk_net_layers', 'modes', 'lr', 'plot_path', 'log_file'],
        'pod-deeponet': ['input_channels_func', 'input_channels_loc', 'output_channels', 'branch_net_layers', 'y_mean_file', 'v_file', 'modes', 'lr', 'plot_path', 'log_file'],
        'geometric-deeponet': ['input_channels_func', 'input_channels_loc', 'output_channels', 'branch_net_layers', 'trunk_net_layers', 'modes', 'lr', 'plot_path', 'log_file'],
        'oformer': ['in_channels', 'out_channels', 'in_emb_dim', 'out_seq_emb_dim', 'heads', 'depth', 'res', 'latent_channels', 'lr', 'plot_path', 'log_file'],
        'lsm': ['in_channels', 'out_channels', 'width', 'num_token', 'num_basis', 'patch_size', 'padding', 'lr', 'plot_path', 'log_file'], 
        'gino': ['n_modes', 'in_channels', 'out_channels', 'hidden_channels', 'projection_channels', 'n_layers', 'lr', 'plot_path', 'log_file'],
        'gnot': ['trunk_size', 'branch_sizes', 'n_layers', 'n_hidden', 'n_head', 'n_inner', 'mlp_layers', 'out_channels', 'in_channels', 'ffn_dropout', 'attn_dropout', 
                 'horiz_fourier_dim', 'lr', 'plot_path', 'log_file'],
    }

    if model_name not in model_params:
        raise ValueError(f"Unknown model name: {model_name}")

    params = {k: v for k, v in config.model.items() if k in model_params[model_name]}

    if model_name == 'fno':
        return FNO(**params)
    elif model_name == 'cno':
        return cno(**params)
    elif model_name == 'uno':
        return UNO(**params)
    elif model_name == 'unet':
        return UNetTrainer(**params)
    elif model_name == 'wno':
        return WNO(**params)
    elif model_name == 'deeponet':
        return DeepONet(**params)
    elif model_name == 'pod-deeponet':
        return PODDeepONet(**params)
    elif model_name == 'geometric-deeponet':
        return GeometricDeepONet(**params)
    elif model_name == 'oformer':
        return OFormer(**params)
    elif model_name == 'lsm':
        return LSMTrainer(**params)   
    elif model_name == 'gnot':
        return GNOT(**params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main(model_name, config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    model = load_model(model_name, config)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    if config.data.type == 'graph':
        test_dataset = LidDrivenGraphDataset(
            file_path_x=config.data.file_path_test_x,
            file_path_y=config.data.file_path_test_y,
            inputs=config.data.inputs
        )
    else:
        test_dataset = LidDrivenDataset(
            file_path_x=config.data.file_path_test_x,
            file_path_y=config.data.file_path_test_y,
            data_type=config.data.type,
            inputs=config.data.inputs
        )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    total_time = 0.0
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            y_true = batch[1].cpu().numpy()
            start_time = time.time()
            y_pred = model(inputs).cpu().numpy()
            end_time = time.time()
            total_time += (end_time - start_time)
            
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

    avg_inference_time = total_time / len(test_loader)
    print(f"Average inference time for {model_name}: {avg_inference_time:.6f} seconds")

    # Create plots directory if it does not exist
    plot_dir = f"plots/{model_name}"
    os.makedirs(plot_dir, exist_ok=True)

    # Save plots
    for i in range(len(all_y_true)):
        plt.figure()
        plt.plot(all_y_true[i].flatten(), label='Actual')
        plt.plot(all_y_pred[i].flatten(), label='Predicted')
        plt.legend()
        plt.title(f"Test Sample {i+1}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.savefig(os.path.join(plot_dir, f"test_sample_{i+1}.png"))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time the inference time of a model.")
    parser.add_argument('--model', type=str, required=True, help='Name of the model to time.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file.')
    args = parser.parse_args()
    main(args.model, args.config, args.checkpoint)

#python3 inference.py --model fno --config /configs/fno/conf.yaml --checkpoint checkpoints/fno/fno-epoch=396-val_loss=0.000000.ckpt