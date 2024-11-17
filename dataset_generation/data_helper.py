import configparser
import numpy as np

def read_time_steps(PATH : str):
    # Read settings from config file (./data.ini)
    config = configparser.ConfigParser()
    config.read(PATH)

    # Read two arrays from config file
    in_time_steps = config.get('time','TIME_STEPS_IN')
    out_time_steps = config.get('time','TIME_STEPS_OUT')

    # Convert to numpy arrays
    in_time_steps = np.array(in_time_steps.split(',')).astype(int)
    out_time_steps = np.array(out_time_steps.split(',')).astype(int)

    return in_time_steps, out_time_steps