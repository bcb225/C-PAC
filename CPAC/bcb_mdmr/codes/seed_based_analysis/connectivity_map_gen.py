import pandas as pd
import numpy as np
import argparse
import os
from nilearn.maskers import NiftiMasker
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from nilearn.image import math_img, load_img
from nilearn.image import resample_to_img

import sys
from scipy.stats import pearsonr, zscore

sys.path.append('../mdmr/')
from DataHandler import DataHandler

# Argument parser for the command-line interface
parser = argparse.ArgumentParser(description='Run CWAS analysis.')
parser.add_argument('--regressor_file', type=str, required=True, help='Regressor file name')
parser.add_argument('--smoothness', type=str, required=True, help='Smoothness of preprocessing')
args = parser.parse_args()

# File directories
mdmr_dir = "/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/"
nas_dir = Path("/mnt/NAS2-2/data/")
MDMR_output_dir = nas_dir / "SAD_gangnam_MDMR"
fmri_prep_dir = nas_dir / "SAD_gangnam_resting_2" / "fMRIPrep_total"
seed_anal_dir = nas_dir / "SAD_gangnam_seed_based_analysis"

# Initialize DataHandler to retrieve group number
data_handler = DataHandler()

# Retrieve the group number for the provided regressor file
group_num = data_handler.get_subject_group(args.regressor_file)
variable_name = data_handler.get_variable(args.regressor_file)

# Print the relevant information for Z-map creation
print(f"Creating Z-map\nSEED: [SMOOTHNESS: {args.smoothness}mm] [GROUP: {group_num}]")

# Load the regressor file into a DataFrame
regressor_df = pd.read_csv(f"{mdmr_dir}/regressor/{args.regressor_file}")
result_dir = MDMR_output_dir / f"{args.smoothness}mm" / str(group_num) / variable_name / "result"
cluster_report_filename = result_dir / "significant_cluster_report.csv"

cluster_report_df = pd.read_csv(cluster_report_filename)
#print(cluster_report_df.columns)
# Ensure "Cluster ID" column is converted to string
cluster_report_df["Cluster ID"] = cluster_report_df["Cluster ID"].astype(str)

# Now you can apply the string method safely
df_numeric_cluster = cluster_report_df[cluster_report_df["Cluster ID"].str.isnumeric()][["Cluster ID", "Center of Mass AAL Label"]]

# Function to process subjects
def process_subject(subject_id, aal_label, cluster_id):
    func_filename = f"{fmri_prep_dir}/sub-{subject_id}/ses-01/func/sub-{subject_id}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed{args.smoothness}mm_resampled4mm_naturebold.nii.gz"
    roi_filename = f"{MDMR_output_dir}/{args.smoothness}mm/{group_num}/{variable_name}/result/cluster_masks/MDMR_significant_aal({aal_label})_label({cluster_id}).nii.gz"
    brain_mask_filename = f"{mdmr_dir}/template/all_final_group_mask_{args.smoothness}mm.nii.gz"
    
    # Load the seed mask (ROI mask) and the brain mask using nilearn
    seed_mask_img = load_img(roi_filename)
    brain_mask_img = load_img(brain_mask_filename)
    seed_mask_img_resampled = resample_to_img(seed_mask_img, brain_mask_img, interpolation='nearest')
    seed_mask_data = seed_mask_img_resampled.get_fdata()
    num_seed_voxels = np.count_nonzero(seed_mask_data)
    #print(f"Number of voxels in the seed mask: {num_seed_voxels}")
    # Create a new brain mask by excluding the seed region using nilearn's math_img
    modified_brain_mask_img = math_img("brain_mask - seed_mask", brain_mask=brain_mask_img, seed_mask=seed_mask_img_resampled)

    # Define the seed masker using the seed mask for extracting time series from the seed region
    seed_masker = NiftiMasker(
        mask_img=seed_mask_img_resampled,
    )
    
    # Define the brain masker using the modified brain mask for whole-brain time series extraction
    brain_masker = NiftiMasker(
        mask_img=brain_mask_img,
    )


    # Extract the seed and brain time series
    seed_time_series = seed_masker.fit_transform(func_filename)  # Shape: (n_timepoints, n_seed_voxels)
    brain_time_series = brain_masker.fit_transform(func_filename)  # Shape: (n_timepoints, n_brain_voxels)

    # Compute the mean seed time series
    mean_seed_time_series = np.mean(seed_time_series, axis=1)  # Shape: (n_timepoints,)

    # Demean the time series
    mean_seed_time_series = mean_seed_time_series - np.mean(mean_seed_time_series)
    brain_time_series = brain_time_series - np.mean(brain_time_series, axis=0)
    """print(f"Seed time series shape: {seed_time_series.shape}")
    print(f"Seed time series mean: {np.mean(seed_time_series)}")
    print(f"Seed time series standard deviation: {np.std(seed_time_series)}")
    print(f"Mean seed time series mean: {np.mean(mean_seed_time_series)}")
    print(f"Mean seed time series standard deviation: {np.std(mean_seed_time_series)}")"""


    # 상관계수를 저장할 배열
    pearson_correlations = np.zeros(brain_time_series.shape[1])  # n_voxels 길이

    # 각 voxel의 시간 시리즈와 seed의 mean signal 간 상관계수 계산
    for voxel_idx in range(brain_time_series.shape[1]):
        voxel_time_series = brain_time_series[:, voxel_idx]  # 각 voxel의 시간 시리즈 (shape: [n_timepoints])
        
        # 피어슨 상관계수 계산
        correlation, p_value = pearsonr(mean_seed_time_series, voxel_time_series)
        
        # 상관계수만 저장 (p-value도 필요하면 함께 저장 가능)
        pearson_correlations[voxel_idx] = correlation

    # 원시 상관계수의 평균 출력
    #raw_mean_pearson_corr = np.mean(pearson_correlations)
    #print(f"Mean of raw Pearson correlations (before Fisher Z-transform): {raw_mean_pearson_corr}")
    # Compute the numerator (covariance between seed and each voxel)
    numerator = np.dot(mean_seed_time_series, brain_time_series)  # Shape: (n_voxels,)

    # Compute the denominator (standard deviations)
    seed_std = np.linalg.norm(mean_seed_time_series)  # Scalar
    brain_std = np.linalg.norm(brain_time_series, axis=0)  # Shape: (n_voxels,)

    # Avoid division by zero
    denominator = seed_std * brain_std
    valid_mask = denominator != 0

    # Compute Pearson correlation coefficients
    seed_to_voxel_correlations = np.zeros(brain_time_series.shape[1])
    seed_to_voxel_correlations[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    seed_to_voxel_correlations_dot = (
        np.dot(brain_time_series.T, mean_seed_time_series) / mean_seed_time_series.shape[0]
    )
    # Convert to Fisher Z
    seed_to_voxel_correlations_fisher_z = np.arctanh(seed_to_voxel_correlations)
    seed_to_voxel_correlations_fisher_z_img = brain_masker.inverse_transform(seed_to_voxel_correlations_fisher_z)

    seed_to_voxel_correlations_fisher_z_dot = np.arctanh(seed_to_voxel_correlations_dot)
    seed_to_voxel_correlations_fisher_z_dot_img = brain_masker.inverse_transform(seed_to_voxel_correlations_fisher_z_dot)
    
    pearson_correlations_fisher_z = np.arctanh(pearson_correlations)
    pearson_correlations_fisher_z_img = brain_masker.inverse_transform(pearson_correlations_fisher_z)
    
    
    # Save the results
    output_dir = f"{seed_anal_dir}/{args.smoothness}mm/corr_z-map/{group_num}/{variable_name}/{aal_label}/"
    os.makedirs(output_dir, exist_ok=True)
    output_dot_dir = f"{seed_anal_dir}/{args.smoothness}mm/corr_dot_z-map/{group_num}/{variable_name}/{aal_label}/"
    os.makedirs(output_dot_dir, exist_ok=True)
    output_pearsonr_dir = f"{seed_anal_dir}/{args.smoothness}mm/corr_pearson_z-map/{group_num}/{variable_name}/{aal_label}/"
    os.makedirs(output_pearsonr_dir, exist_ok=True)
    
    output_filename = f"{output_dir}/sub-{subject_id}_fisher_z_img.nii.gz"
    seed_to_voxel_correlations_fisher_z_img.to_filename(output_filename)
    output_dot_filename = f"{output_dot_dir}/sub-{subject_id}_dot_fisher_z_img.nii.gz"
    seed_to_voxel_correlations_fisher_z_dot_img.to_filename(output_dot_filename)
    output_pearsonr_filename = f"{output_pearsonr_dir}/sub-{subject_id}_pearsonr_fisher_z_img.nii.gz"
    pearson_correlations_fisher_z_img.to_filename(output_pearsonr_filename)

    """# Fisher Z 변환된 값들의 평균 출력
    print(f"Mean of fisher z_dot: {np.mean(seed_to_voxel_correlations_fisher_z_dot)}")  # seed_to_voxel_correlations_dot의 평균 확인
    print(f"Mean of fisher z: {np.mean(seed_to_voxel_correlations_fisher_z)}")  # seed_to_voxel_correlations의 평균 확인
    print(f"Mean of fisher z (pearson_correlations): {np.mean(pearson_correlations_fisher_z)}")  # pearson_correlations의 평균 확인

    # Fisher Z 변환된 값들의 표준편차 출력
    print(f"Standard deviation of fisher z_dot: {np.std(seed_to_voxel_correlations_fisher_z_dot)}")  # seed_to_voxel_correlations_dot의 표준편차 확인
    print(f"Standard deviation of fisher z: {np.std(seed_to_voxel_correlations_fisher_z)}")  # seed_to_voxel_correlations의 표준편차 확인
    print(f"Standard deviation of fisher z (pearson_correlations): {np.std(pearson_correlations_fisher_z)}")  # pearson_correlations의 표준편차 확인"""
    return f"Processed {subject_id}, saved {output_filename}"
# Wrapper function to be passed to multiprocessing (cannot use lambda)
def wrapper_process_subject(args):
    return process_subject(*args)

if __name__ == '__main__':
    for index, row in df_numeric_cluster.iterrows():
        cluster_id = row['Cluster ID']
        aal_label = row['Center of Mass AAL Label']
        print(f"Processing Cluster ID = {cluster_id}, AAL Label = {aal_label}")
        
        subject_id_list = [subdir.name.replace('sub-', '') for subdir in fmri_prep_dir.iterdir() if subdir.is_dir() and subdir.name.startswith('sub-')]

        # Use multiprocessing to process subjects in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Use tqdm to monitor progress
            with tqdm(total=len(subject_id_list), desc=f"Processing Cluster {cluster_id}") as pbar:
                args_list = [(subject_id, aal_label, cluster_id) for subject_id in subject_id_list]
                for _ in pool.imap_unordered(wrapper_process_subject, args_list):
                    pbar.update()
