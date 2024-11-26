import os
import yaml

# Define the path to the 'configs' directory
config_dir = "/work/mech-ai/edherron/flowbench-sciml/configs/ns"

# Iterate through each subdirectory in 'configs'
for subdir in os.listdir(config_dir):
    subdir_path = os.path.join(config_dir, subdir)
    
    # Check if the directory contains a conf.yaml file
    conf_path = os.path.join(subdir_path, "conf.yaml")
    if os.path.isfile(conf_path):
        # Load the YAML file
        with open(conf_path, "r") as file:
            config = yaml.safe_load(file)
        
        # Check if the 'data' section exists
        if 'callbacks' in config:
            config['callbacks']['checkpoint']['every_n_epochs'] = 100
            config['callbacks']['checkpoint']['dirpath'] = '/work/mech-ai/edherron/flowbench-sciml/checkpoints/ns/512_sdf_easy/'
            config['callbacks']['checkpoint']['save_top_k'] = -1
            # if config['callbacks']['dirpath'] is not None:
            #     del config['callbacks']['dirpath']
            
        if 'trainer' in config:
            config['trainer']['max_epochs'] = 2
        
        if 'data' in config:
            config['data']['inputs'] = 'sdf'
            config['data']['file_path_train_x'] = '/work/mech-ai-scratch/arabeh/data_prep_ns/LDC_NS_2D/processed/easy/harmonics_ldc_train_x.npz'
            config['data']['file_path_train_y'] = '/work/mech-ai-scratch/arabeh/data_prep_ns/LDC_NS_2D/processed/easy/harmonics_ldc_train_y.npz'
            config['data']['file_path_test_x'] = '/work/mech-ai-scratch/arabeh/data_prep_ns/LDC_NS_2D/processed/easy/harmonics_ldc_test_x.npz'
            config['data']['file_path_test_y'] = '/work/mech-ai-scratch/arabeh/data_prep_ns/LDC_NS_2D/processed/easy/harmonics_ldc_test_y.npz'
            config['data']['equation'] = 'ns'
            
        if 'sweep_parameters' in config:
            config['sweep_parameters']['seed']['values'] = [0, 10, 20]
            
        if 'model' in config:
            config['model']['out_channels'] = 3
            # if config['model']['output_channels'] is not None:
            #     del config['model']['output_channels']
        
            
            # Write the updated YAML file
            with open(conf_path, "w") as file:
                yaml.dump(config, file, default_flow_style=False)

        print(f"Updated: {conf_path}")
    else:
        print(f"Skipped: {subdir} (no conf.yaml found)")
