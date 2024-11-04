import sys
import os
import argparse
import nibabel as nb
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import t
from numpy import inf

from CPAC.cwas.mdmr import mdmr
from CPAC.utils import correlation

from CPAC.pipeline.cpac_ga_model_generator import (create_merge_mask,
                                                   create_merged_copefile)
from CPAC.cwas.cwas import *
import nibabel as nib

sys.path.append('~/fmri_project/C-PAC/CPAC/bcb_mdmr/codes/mdmr/')
from DataHandler import DataHandler
from Telegram import Telegram  # Import the modified Telegram class

# Set up argument parser
parser = argparse.ArgumentParser(description='Run CWAS analysis.')
parser.add_argument('--regressor_file', type=str, required=True, help="Regressor filename (csv)")
parser.add_argument('--smoothness', type=str, required=True, help='Smoothness of preprocessing')
parser.add_argument('--mode', type=str, required=True, help='Mode of preprocessing')

args = parser.parse_args()

# Get parameters from args
regressor_file = args.regressor_file
smoothness = args.smoothness
data_handler = DataHandler()
variable_of_interest = data_handler.get_variable(regressor_file)
subjects = data_handler.subject_dict_maker(regressor_file, smoothness)
subject_group = data_handler.get_subject_group(regressor_file)
target_subject_index = False
mask_file = data_handler.get_mask_file(smoothness)
regressor_file = data_handler.get_regressor_file(regressor_file)
participant_column = "Participant"
columns_string = variable_of_interest
mode = args.mode
permutations = 15000
voxel_range = data_handler.get_voxel_range(smoothness)

# Initialize the Telegram class
telegram = Telegram()

# Prepare the start message
start_message = (f"Starting MDMR Process\n"
                 f"Regressor File: {args.regressor_file}\n"
                 f"Subject Group Index: {subject_group}\n"
                 f"Variable of Interest: {variable_of_interest}\n"
                 f"Smoothness: {args.smoothness}")

# Print and send the start message
print(start_message)
telegram.send_msg(start_message)

# Continue with the MDMR analysis
temp_dir, voxel_range = nifti_cwas(subjects, mask_file, regressor_file, participant_column,
                columns_string, permutations, variable_of_interest, smoothness, voxel_range, subject_group, target_subject_index, mode)

cwas_batches = [
    (temp_dir, voxel_range)
]

z_score = [1]  # if you want to compute Z-scores
F_file, p_file, log_p_file, one_p_file, z_file = merge_cwas_batches(cwas_batches, mask_file, z_score, permutations, subject_group, variable_of_interest, smoothness)

# Prepare the result message
result_message = (f"Finished MDMR Process\n"
                  f"Regressor File: {args.regressor_file}\n"
                  f"Subject Group Index: {subject_group}\n"
                  f"Variable of Interest: {variable_of_interest}\n"
                  f"Smoothness: {args.smoothness}\n"
                  )

# Print and send the result message
print(result_message)
telegram.send_msg(result_message)

# Output the result file paths
print("Results saved to:")
print(F_file)
print(p_file)
print(log_p_file)
print(one_p_file)
print(z_file)
