import configparser
import numpy as np
import time
import logging
import glob

def read_time_steps(PATH : str):
    # Read time settings from config file (./data.ini)
    config = configparser.ConfigParser()
    config.read(PATH)

    # Read two arrays from config file
    in_time_steps = config.get('time','TIME_STEPS_IN')
    out_time_steps = config.get('time','TIME_STEPS_OUT')

    # Convert to numpy arrays
    in_time_steps = np.array(in_time_steps.split(',')).astype(int)
    out_time_steps = np.array(out_time_steps.split(',')).astype(int)

    return in_time_steps, out_time_steps

def read_data_settings(PATH : str):
    # Read dataset settings from config file (./data.ini)
    config = configparser.ConfigParser()
    config.read(PATH)

    # Read two arrays from config file
    num_samples = config.get('dataset','PCT_SAMPLES')
    res_x = config.get('dataset','RESOLUTION_X')
    res_y = config.get('dataset','RESOLUTION_Y')
    root_data_dir = config.get('dataset','ROOT_DATA_DIR')
    root_geometry_dir = config.get('dataset','ROOT_GEOMETRY_DIR')

    return float(num_samples), int(res_x), int(res_y), root_data_dir, root_geometry_dir


def sample_counter(root_data_dir : str):
    # Count the number of samples in the dataset
    structure = root_data_dir + '/**/Re_*.npz'
    samps = 0
    for file in glob.glob(structure, recursive=True):
        samp += 1
    return samps

# Write a decorator function that logs the time taken by a function to execute
def log_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f'{func.__name__} took {end-start} seconds to execute')
        return result

@log_time
def data_download(root_data_dir : str, root_geometry_dir : str, res_x : int, res_y : int, in_time_steps : list, out_time_steps : list):
    
    #Get the number of samples, resolution of the data, and the root directories for data and geometry
    num_samples = sample_counter(root_data_dir)
    
    #Define the Input and Output Channels
    in_channels = 3*len(in_time_steps) + 2 # 3 channels for each time step and 2 channels for the geometry + Reynolds number
    out_channels = 3*len(out_time_steps)
    
    #Initialize the numpy arrays for input and output data. Allot memory for the numpy arrays.
    in_data = np.zeros((num_samples, in_channels, res_y, res_x))
    out_data = np.zeros((num_samples, out_channels, res_y, res_x))
    
    samp = 0 #Track the number of samples
    
    # Read data from npz files for the specified time steps. Load the data into the numpy arrays in_data and out_data.
    
    #Iterate over the folder structure to read the data
    structure = root_data_dir + '/**/Re_*.npz'
    
    #Extract the geometry and reynolds number from the path of the npz file
    for file in glob.glob(structure, recursive=True):
        
        #Read the geometry and reynolds number from the path
        sdf_data = file.split('/')[-2]
        reynolds_number = file.split('/')[-1]
        reynolds_number = reynolds_number.split('_')[-1]
        reynolds_number = reynolds_number.split('.')[0]
        
        #Load the geometry data
        geometry_data = np.load(root_geometry_dir + '/sdf_'+sdf_data+'.npz')['data']
        #Scale the geometry data to the resolution of the flow data
        scale = geometry_data.shape[1] // res_x
        geometry_data = geometry_data[::scale, ::scale]
        
        #Load this geometry data to in_data
        in_data[samp, 0, :, :] = geometry_data
        
        #Create the reynolds number field
        reynolds_number = np.ones((res_y, res_x)) * int(reynolds_number)
        
        #Load this reynolds number to in_data
        in_data[samp, 1, :, :] = reynolds_number
        
        #Read the flow data
        flow_data = np.load(file)['data']
        
        #Select the time steps for input and output data
        flow_data_in = flow_data[in_time_steps, :, :, :]
        flow_data_out = flow_data[out_time_steps, :, :, :]
        
        #Move the last dimension to the second dimension
        flow_data_in = flow_data_in.transpose(0,3,1,2)
        flow_data_out = flow_data_out.transpose(0,3,1,2)
        
        #Combine the first two dimensions to get the input and output channels
        flow_data_in = flow_data_in.reshape(-1, res_y, res_x)
        flow_data_out = flow_data_out.reshape(-1, res_y, res_x)
        
        #Load the flow data to in_data and out_data
        in_data[samp, 2:, :, :] = flow_data_in
        out_data[samp, :, :, :] = flow_data_out
        
        samp += 1
    
    print("Total number of processed samples = ", samp)
    
    #Return the input and output data 
    return in_data, out_data


