import numpy as np
import configparser

if __name__ == '__main__':
   # Read settings from config file (./data.ini)
    config = configparser.ConfigParser()
    config.read('data.ini')
    
    print(config.sections())
    
    # Read two arrays from config file
    in_time_steps = config.get('time','TIME_STEPS_IN')
    out_time_steps = config.get('time','TIME_STEPS_OUT')
    
    # Convert to numpy arrays
    in_time_steps = np.array(in_time_steps.split(',')).astype(np.int)
    out_time_steps = np.array(out_time_steps.split(',')).astype(np.int)
    
    # Print the arrays
    print(in_time_steps)
    print(out_time_steps)