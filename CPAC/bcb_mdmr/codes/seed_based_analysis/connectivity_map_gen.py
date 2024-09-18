import pandas as pd
import numpy as np
import argparse
import os
from nilearn.maskers import NiftiMasker
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
parser = argparse.ArgumentParser(description='Run CWAS analysis.')
parser.add_argument('--group', type=str, required=True, help='Subject group name')
parser.add_argument('--variable', type=str, required=True, help='Variable of interest')
parser.add_argument('--smoothness', type=str, required=True, help='Smoothness of preprocessing')
args = parser.parse_args()

mdmr_dir = "/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/"
nas_dir = "/mnt/NAS2-2/data/"
MDMR_output_dir = f"{nas_dir}/SAD_gangnam_MDMR/"
fmri_prep_dir = f"{nas_dir}/SAD_gangnam_resting_2/fMRIPrep_total"
seed_anal_dir = f"{nas_dir}/SAD_gangnam_seed_based_analysis/"

regressor_df = pd.read_csv(
    f"{mdmr_dir}/input/{args.group}_{args.variable}_regressor.csv"
)

print(f"Creating Z-map\nSEED: [SMOOTHNESS: {args.smoothness}mm] [GROUP: {args.group}] [VARIABLE: {args.variable}]")

subject_id_list = regressor_df["Participant"].values
for subject_id in tqdm(subject_id_list, desc="Processing Subjects"):
    seed_masker = NiftiMasker(
        mask_img = f"{MDMR_output_dir}/{args.smoothness}mm/{args.group}/{args.variable}/result/significant_cluster.nii.gz",
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
    )
    func_filename = f"{fmri_prep_dir}/sub-{subject_id}/ses-01/func/sub-{subject_id}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed6mm_resampled4mm_bold.nii.gz"
    #confound_filename = f"{seed_anal_dir}/modified_confounds/sub-{subject_id}_ses-01_task-rest_desc-confounds_timeseries_modified.tsv"
    
    seed_time_series = seed_masker.fit_transform(
       func_filename,#confounds=[confound_filename]
    )
    
    brain_masker = NiftiMasker(
        mask_img = f"{mdmr_dir}/template/gangnam_total_final_group_mask_{args.smoothness}mm.nii.gz",
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
    )
    brain_time_series = brain_masker.fit_transform(
        func_filename,#confounds=[confound_filename]
    )
    mean_seed_time_series = np.mean(seed_time_series, axis=1)
    """seed_to_voxel_correlations = (
        np.dot(brain_time_series.T, mean_seed_time_series) / mean_seed_time_series.shape[0]
    )"""
    seed_to_voxel_correlations = (
        np.dot(brain_time_series.T, mean_seed_time_series) / mean_seed_time_series.shape[0]
    )    

    seed_to_voxel_correlations_fisher_z = np.arctanh(seed_to_voxel_correlations)

    seed_to_voxel_correlations_fisher_z_img = brain_masker.inverse_transform(
        seed_to_voxel_correlations_fisher_z.T
    )
    os.makedirs(f"{seed_anal_dir}/{args.smoothness}mm/corr_z-map/seed_{args.group}_{args.variable}", exist_ok=True)
    seed_to_voxel_correlations_fisher_z_img.to_filename(
        f"{seed_anal_dir}/{args.smoothness}mm/corr_z-map/seed_{args.group}_{args.variable}/sub-{subject_id}_fisher_z_img.nii.gz"
    )
