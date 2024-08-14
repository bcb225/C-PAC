import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
from DataHandler import DataHandler

subject_group = "gangnam_total"
variable_of_interest = "LSAS"
data_handler = DataHandler()
subjects = data_handler.subject_dict_maker(subject_group)
target_subject_index = False
mask_file = data_handler.get_mask_file(subject_group)
regressor_file = data_handler.get_regressor_file(subject_group, variable_of_interest)
participant_column = "Participant"
columns_string = variable_of_interest
permutations = 15000
voxel_range = data_handler.get_voxel_range(subject_group)


F_file, p_file, voxel_range = nifti_cwas(subjects, mask_file, regressor_file, participant_column,
               columns_string, permutations, voxel_range, subject_group, target_subject_index)

cwas_batches = [
    (F_file, p_file, voxel_range)
]

z_score = [1]  # if you want to compute Z-scores
F_file, p_file, log_p_file, one_p_file, z_file = merge_cwas_batches(cwas_batches, mask_file, z_score, permutations, subject_group,variable_of_interest)

print("Results saved to:")
print(F_file)
print(p_file)
print(log_p_file)
print(one_p_file)
print(z_file)