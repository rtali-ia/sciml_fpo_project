import numpy as np
from data_helper import read_time_steps, read_data_settings, data_download
import logging
from data_logger import setup_logger

if __name__ == '__main__':
    
    # Init logger
    log_file_name = 'data_prep.log'
    logger = setup_logger(log_file_name, level=logging.INFO)
    
   # Read settings from config file (./data.ini)
    INI_FILE_PATH = '/work/mech-ai/rtali/projects/sciml_fpo_project/dataset_generation/data.ini'
    in_time_steps, out_time_steps = read_time_steps(INI_FILE_PATH)
    pct_samples, res_x, res_y, root_data_dir, root_geometry_dir = read_data_settings(INI_FILE_PATH)

    logger.info('Read all settings from config file - Done')
    
    '''
    <root_dir>
        <geometry>
            <Reynolds_number>/solution.npz (The npz files has 240 time steps, 3 channels (u, v, p) and 2048 x 512)
            ...
            ...
            ...
            ...
        ...
            ...
            ...
            ...
            ...
            ...
    '''
    
    #Wait for the user to press a key to continue
    input('Check Logs and Press Enter to continue .........')
    
    # Download data from the root directories
    in_data, out_data = data_download(root_data_dir, root_geometry_dir, res_x, res_y, in_time_steps, out_time_steps)
    
    logger.info('Read all data from npz files - Done')
    
    #Subset the data based on the percentage of samples specified in the config file (./data.ini) randomly
    #Choose the indices randomly
    total_samples = in_data.shape[0]
    num_samples = int(pct_samples*total_samples)
    indices = np.random.choice(total_samples, num_samples, replace=False)
    
    #Subset the data
    in_data = in_data[indices]
    out_data = out_data[indices]
    
    logger.info('Subset the data based on the percentage of samples specified in the config file - Done')
    
    #Save the data to npz files
    np.savez_compressed('./dataset_generation/fpo/in_data.npz', in_data)
    np.savez_compressed('./dataset_generation/fpo/out_data.npz', out_data)
    
    logger.info('Saved the sampled data to npz files - Done')