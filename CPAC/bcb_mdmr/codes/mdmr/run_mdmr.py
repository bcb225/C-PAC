import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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

# Set up argument parser
parser = argparse.ArgumentParser(description='Run CWAS analysis.')
parser.add_argument('--group', type=str, required=True, help='Subject group name')
parser.add_argument('--variable', type=str, required=True, help='Variable of interest')

args = parser.parse_args()

# Get parameters from args
subject_group = args.group
variable_of_interest = args.variable

data_handler = DataHandler()
subjects = data_handler.subject_dict_maker(subject_group)
target_subject_index = False
mask_file = data_handler.get_mask_file(subject_group)
regressor_file = data_handler.get_regressor_file(subject_group, variable_of_interest)
participant_column = "Participant"
columns_string = variable_of_interest
permutations = 15000
voxel_range = data_handler.get_voxel_range(subject_group)


temp_dir, voxel_range = nifti_cwas(subjects, mask_file, regressor_file, participant_column,
               columns_string, permutations, variable_of_interest, voxel_range, subject_group, target_subject_index)

cwas_batches = [
    (temp_dir, voxel_range)
]

z_score = [1]  # if you want to compute Z-scores
F_file, p_file, log_p_file, one_p_file, z_file = merge_cwas_batches(cwas_batches, mask_file, z_score, permutations, subject_group,variable_of_interest)

print("Results saved to:")
print(F_file)
print(p_file)
print(log_p_file)
print(one_p_file)
print(z_file)