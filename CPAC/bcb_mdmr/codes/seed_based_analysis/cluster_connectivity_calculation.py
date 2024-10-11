import pandas as pd
import numpy as np
import argparse
import os
from nilearn.maskers import NiftiMasker
from pathlib import Path
import sys
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
sys.path.append('../mdmr/')
from DataHandler import DataHandler

# Set the base directory for NAS
nas_dir = Path("/mnt/NAS2-2/data/")

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run CWAS analysis.')
    parser.add_argument('--regressor_file', type=str, required=True, help='Regressor file name')
    parser.add_argument('--smoothness', type=str, required=True, help='Smoothness of preprocessing')
    parser.add_argument('--mode', type=str, required=False, help='Mode of operation, e.g., regress', default="regress")
    return parser.parse_args()

def check_file_exists(file_path):
    # Check if a file exists, raise an error if it does not
    if not Path(file_path).exists():
        raise FileNotFoundError(f"{file_path} does not exist.")

from scipy.stats import pearsonr

def corr_calc(subject_id, source_mask, target_mask):
    # Directory for fMRI preprocessed data
    fmri_prep_dir = nas_dir / "SAD_gangnam_resting_2" / "fMRIPrep_total"
    func_filename = f"{fmri_prep_dir}/sub-{subject_id}/ses-01/func/sub-{subject_id}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed{args.smoothness}mm_resampled4mm_bold.nii.gz"
    
    # Check if the functional file exists
    check_file_exists(func_filename)
    
    # Create NiftiMaskers for the source and target masks, with standardization
    source_masker = NiftiMasker(
        mask_img=source_mask,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample"
    )
    target_masker = NiftiMasker(
        mask_img=target_mask,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample"
    )
    
    # Extract time series data from the functional file for both source and target masks
    source_time_series = source_masker.fit_transform(func_filename)  # Shape: (n_timepoints, n_source_voxels)
    target_time_series = target_masker.fit_transform(func_filename)  # Shape: (n_timepoints, n_target_voxels)

    # Compute the mean time series for both source and target
    mean_source_time_series = np.mean(source_time_series, axis=1)  # Shape: (n_timepoints,)
    mean_target_time_series = np.mean(target_time_series, axis=1)  # Shape: (n_timepoints,)

    # Compute the Pearson correlation coefficient
    correlation, p_value = pearsonr(mean_source_time_series, mean_target_time_series)

    # Handle potential numerical issues with arctanh
    epsilon = 1e-7
    correlation = np.clip(correlation, -1 + epsilon, 1 - epsilon)

    # Convert correlation to Fisher Z values for normalization
    seed_to_voxel_correlations_fisher_z = np.arctanh(correlation)

    return seed_to_voxel_correlations_fisher_z

def process_participant(participant, source_cluster_mask, significant_cluster_mask, peak_cluster_mask, center_cluster_mask):
    # Calculate correlations for different cluster masks
    significant_correlation = corr_calc(participant, source_cluster_mask, significant_cluster_mask)
    peak_correlation = corr_calc(participant, source_cluster_mask, peak_cluster_mask)
    center_correlation = corr_calc(participant, source_cluster_mask, center_cluster_mask)

    # Return the results in a dictionary
    return {
        "participant": participant,
        "significant_correlation": significant_correlation,
        "peak_correlation": peak_correlation,
        "center_correlation": center_correlation
    }

def process_participant_wrapper(args):
    participant, source_cluster_mask, significant_cluster_mask, peak_cluster_mask, center_cluster_mask = args
    return process_participant(participant, source_cluster_mask, significant_cluster_mask, peak_cluster_mask, center_cluster_mask)

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()

    # File directories
    mdmr_dir = "/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/"
    MDMR_output_dir = nas_dir / "SAD_gangnam_MDMR"
    seed_anal_dir = nas_dir / "SAD_gangnam_seed_based_analysis"

    # Initialize DataHandler to retrieve group number
    data_handler = DataHandler()
    regressor_dir = Path(mdmr_dir) / "regressor"
    
    # Retrieve group number and variable name for the provided regressor file
    group_num = data_handler.get_subject_group(args.regressor_file)
    source_variable_name = data_handler.get_variable(args.regressor_file)

    # Load the regressor DataFrame
    regressor_df = pd.read_csv(f"{mdmr_dir}/regressor/{args.regressor_file}")
    result_dir = MDMR_output_dir / f"{args.smoothness}mm" / str(group_num) / source_variable_name / "result"
    source_cluster_report_filename = result_dir / "significant_cluster_report.csv"

    # Check if the cluster report file exists
    check_file_exists(source_cluster_report_filename)

    # Load cluster report and filter numeric cluster IDs
    source_cluster_report_df = pd.read_csv(source_cluster_report_filename)
    source_cluster_report_df["Cluster ID"] = source_cluster_report_df["Cluster ID"].astype(str)

    source_f_numeric_cluster = source_cluster_report_df[source_cluster_report_df["Cluster ID"].str.isnumeric()][["Cluster ID", "Center of Mass AAL Label"]]

    # Check the mode
    if args.mode == 'regress':
        # Loop through all regressor files and only include those with the same variable name
        all_regressor_files = list(regressor_dir.glob("*_regressor.csv"))
        matching_regressor_files = []

        for target_regressor_file_path in all_regressor_files:
            # Extract the variable name from each regressor file
            other_variable_name = data_handler.get_variable(target_regressor_file_path.name)

            # Only include files with the same variable name
            if other_variable_name == source_variable_name:
                matching_regressor_files.append(target_regressor_file_path)

        # Replace all_regressor_files with the filtered list
        all_regressor_files = matching_regressor_files

    else:
        # Original behavior: Use all available regressor files, excluding a specific pseudo variable file
        all_regressor_files = list(regressor_dir.glob("*_regressor.csv"))
        all_regressor_files = [file for file in all_regressor_files if "all_pseudo_variable_regressor.csv" not in file.name]
    
    all_participant_regressor = pd.read_csv(regressor_dir / "all_pseudo_variable_regressor.csv")
    participant_list = all_participant_regressor["Participant"].values

    # Iterate through all matching regressor files
    for target_regressor_file_path in tqdm(all_regressor_files):
        target_regressor_file = target_regressor_file_path.name
        target_variable_name = data_handler.get_variable(target_regressor_file)
        target_regressor_name = target_regressor_file.replace("_regressor.csv", "")
        target_regressor_df = pd.read_csv(target_regressor_file_path)
        
        # Iterate through each source cluster
        for index, row in source_f_numeric_cluster.iterrows():
            source_aal_label = row['Center of Mass AAL Label']
            source_cluster_id = row['Cluster ID']
            
            # Define file paths for cluster tables and cluster masks
            target_cluster_table_file =  seed_anal_dir / f"{args.smoothness}mm" / "second_level_results" / str(group_num) / source_variable_name / source_aal_label / target_regressor_file_path.stem / f"{source_aal_label}_cluster_table.csv"
            try:
                check_file_exists(target_cluster_table_file)
                target_cluster_table = pd.read_csv(target_cluster_table_file)
            except FileNotFoundError as e:
                print(e)
                continue

            target_cluster_table["Cluster ID"] = target_cluster_table["Cluster ID"].astype(str)
            target_numeric_cluster = target_cluster_table[target_cluster_table["Cluster ID"].str.isnumeric()][["Cluster ID", "Center of Mass AAL Label"]]
            
            # Iterate through each target cluster
            for target_index, target_row in target_numeric_cluster.iterrows():
                target_aal_label = target_row['Center of Mass AAL Label']
                target_cluster_id = target_row['Cluster ID']
                
                # Define paths to different types of cluster masks
                significant_cluster_mask = seed_anal_dir / f"{args.smoothness}mm" / "second_level_results" / str(group_num) / source_variable_name / source_aal_label / target_regressor_file_path.stem / "cluster_masks" / f"significant_aal_{target_aal_label}_label_{target_cluster_id}.nii.gz"
                peak_cluster_mask = seed_anal_dir / f"{args.smoothness}mm" / "second_level_results" / str(group_num) / source_variable_name / source_aal_label / target_regressor_file_path.stem / "cluster_masks" / f"peak_stat_sphere_aal_{target_aal_label}_label_{target_cluster_id}.nii.gz"
                center_cluster_mask = seed_anal_dir / f"{args.smoothness}mm" / "second_level_results" / str(group_num) / source_variable_name / source_aal_label / target_regressor_file_path.stem / "cluster_masks" / f"center_of_mass_sphere_aal_{target_aal_label}_label_{target_cluster_id}.nii.gz"
                source_cluster_mask = f"{MDMR_output_dir}/{args.smoothness}mm/{group_num}/{source_variable_name}/result/cluster_masks/MDMR_significant_aal({source_aal_label})_label({source_cluster_id}).nii.gz"

                # Check if all cluster masks exist
                try:
                    check_file_exists(significant_cluster_mask)
                    check_file_exists(peak_cluster_mask)
                    check_file_exists(center_cluster_mask)
                    check_file_exists(source_cluster_mask)
                except FileNotFoundError as e:
                    print(e)
                    continue

                # Use multiprocessing to calculate correlations for each participant
                with Pool(cpu_count()) as pool:
                    results = list(tqdm(pool.imap(process_participant_wrapper, [(participant, source_cluster_mask, significant_cluster_mask, peak_cluster_mask, center_cluster_mask) for participant in participant_list]), total=len(participant_list)))

                # Add correlations to the target_regressor DataFrame and save it
                result_df = all_participant_regressor.copy()
                result_df["Source Regressor"] = source_variable_name
                result_df["Target Regressor"] = target_variable_name
                result_df["Significant Correlation"] = [res['significant_correlation'] for res in results]
                result_df["Peak Correlation"] = [res['peak_correlation'] for res in results]
                result_df["Center Correlation"] = [res['center_correlation'] for res in results]

                # Define the output file path
                stat_value = target_cluster_table.loc[target_cluster_table['Cluster ID'] == target_cluster_id, 'Peak Stat'].values[0]
                stat_value_rounded = round(stat_value, 2)
                output_file_path = seed_anal_dir / f"{args.smoothness}mm" / "second_level_results" / str(group_num) / source_variable_name / source_aal_label / target_regressor_file_path.stem / "cluster_masks" / f"{source_aal_label}_{target_aal_label}_{stat_value_rounded}.csv"

                # Save the result DataFrame to CSV
                result_df.to_csv(output_file_path, index=False)