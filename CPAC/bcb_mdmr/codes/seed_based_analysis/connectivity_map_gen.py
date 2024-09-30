import pandas as pd
import numpy as np
import argparse
import os
from nilearn.maskers import NiftiMasker
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
import sys
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
df_numeric_cluster = cluster_report_df[cluster_report_df["Cluster ID"].str.isnumeric()][["Cluster ID", "AAL Label"]]

# Function to process subjects
def process_subject(subject_id, aal_label, cluster_id):
    func_filename = f"{fmri_prep_dir}/sub-{subject_id}/ses-01/func/sub-{subject_id}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed6mm_resampled4mm_bold.nii.gz"
    roi_filename = f"{MDMR_output_dir}/{args.smoothness}mm/{group_num}/{variable_name}/result/MDMR_significant_aal({aal_label})_label({cluster_id}).nii.gz"
    
    seed_masker = NiftiMasker(
        mask_img=roi_filename,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample"
    )
    
    brain_masker = NiftiMasker(
        mask_img=f"{mdmr_dir}/template/all_final_group_mask_{args.smoothness}mm.nii.gz",
        standardize="zscore_sample",
        standardize_confounds="zscore_sample"
    )

    # Extract the seed and brain time series
    seed_time_series = seed_masker.fit_transform(func_filename)
    brain_time_series = brain_masker.fit_transform(func_filename)

    # Compute the seed-to-voxel correlations
    mean_seed_time_series = np.mean(seed_time_series, axis=1)
    seed_to_voxel_correlations = (
        np.dot(brain_time_series.T, mean_seed_time_series) / mean_seed_time_series.shape[0]
    )
    
    # Convert to Fisher Z
    seed_to_voxel_correlations_fisher_z = np.arctanh(seed_to_voxel_correlations)
    seed_to_voxel_correlations_fisher_z_img = brain_masker.inverse_transform(seed_to_voxel_correlations_fisher_z.T)

    # Save the results
    output_dir = f"{seed_anal_dir}/{args.smoothness}mm/corr_z-map/{group_num}/{variable_name}/{aal_label}/"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/sub-{subject_id}_fisher_z_img.nii.gz"
    seed_to_voxel_correlations_fisher_z_img.to_filename(output_filename)
    return f"Processed {subject_id}, saved {output_filename}"

# Wrapper function to be passed to multiprocessing (cannot use lambda)
def wrapper_process_subject(args):
    return process_subject(*args)

if __name__ == '__main__':
    for index, row in df_numeric_cluster.iterrows():
        cluster_id = row['Cluster ID']
        aal_label = row['AAL Label']
        print(f"Processing Cluster ID = {cluster_id}, AAL Label = {aal_label}")
        
        subject_id_list = regressor_df["Participant"].values

        # Use multiprocessing to process subjects in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Use tqdm to monitor progress
            with tqdm(total=len(subject_id_list), desc=f"Processing Cluster {cluster_id}") as pbar:
                args_list = [(subject_id, aal_label, cluster_id) for subject_id in subject_id_list]
                for _ in pool.imap_unordered(wrapper_process_subject, args_list):
                    pbar.update()
