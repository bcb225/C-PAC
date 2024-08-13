import os

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

subjects = {
    "c0062": "/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep/sub-c0062/ses-01/func/sub-c0062_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "c0290": "/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep/sub-c0290/ses-01/func/sub-c0290_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "c0010": "/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep/sub-c0010/ses-01/func/sub-c0010_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "c0024": "/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep/sub-c0024/ses-01/func/sub-c0024_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "c0025": "/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep/sub-c0025/ses-01/func/sub-c0025_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "c0028": "/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep/sub-c0028/ses-01/func/sub-c0028_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "s0250": "/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep/sub-s0250/ses-01/func/sub-s0250_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
}

mask_file = "/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/template/sample_small_final_group_mask.nii.gz"

regressor_file = "/home/changbae/fmri_project/czarrar/input_files/model_evs.csv"

participant_column = "Participant"

columns_string = "LSAS_performance"

permutations = 15000

voxel_range = np.arange(0, 102)

F_file, p_file, voxel_range = nifti_cwas(subjects, mask_file, regressor_file, participant_column,
               columns_string, permutations, voxel_range)

cwas_batches = [
    (F_file, p_file, voxel_range)
]

z_score = [1]  # if you want to compute Z-scores
F_file, p_file, log_p_file, one_p_file, z_file = merge_cwas_batches(cwas_batches, mask_file, z_score, permutations)

print("Results saved to:")
print(F_file)
print(p_file)
print(log_p_file)
print(one_p_file)
print(z_file)