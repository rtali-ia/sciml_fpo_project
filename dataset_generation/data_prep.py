import numpy as np
import configparser
from data_helper import read_time_steps

if __name__ == '__main__':
   # Read settings from config file (./data.ini)
    INI_FILE_PATH = '/work/mech-ai/rtali/projects/sciml_fpo_project/dataset_generation/data.ini'
    in_time_steps, out_time_steps = read_time_steps(INI_FILE_PATH)
    
    print(in_time_steps)
    print(out_time_steps)