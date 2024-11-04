import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import datasets
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.image import load_img, resample_to_img
from scipy.stats import pearsonr
import argparse
import os
import sys
import multiprocessing as mp
from tqdm import tqdm

# Argument parser for the command-line interface
parser = argparse.ArgumentParser(description='Run functional connectivity analysis.')
parser.add_argument('--regressor_file', type=str, required=True, help='Regressor file name')
parser.add_argument('--smoothness', type=str, required=True, help='Smoothness of preprocessing')
args = parser.parse_args()

# Append the custom module path if needed
sys.path.append('../mdmr/')
from DataHandler import DataHandler

# Define your parameters and directories

# NAS directory
nas_dir = Path("/mnt/NAS2-2/data/")

# fmri_prep directory
fmri_prep_dir = nas_dir / "SAD_gangnam_resting_2" / "fMRIPrep_total"

# Participant list
participant_list = [subdir.name.replace('sub-', '') for subdir in fmri_prep_dir.iterdir() if subdir.is_dir() and subdir.name.startswith('sub-')]

# MDMR output directory
MDMR_output_dir = nas_dir / "SAD_gangnam_MDMR"

# MDMR directory
mdmr_dir = "/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/"

# Smoothness value
args_smoothness = args.smoothness

# Initialize DataHandler to retrieve group number and variable name
data_handler = DataHandler()
group_num = data_handler.get_subject_group(args.regressor_file)
variable_name = data_handler.get_variable(args.regressor_file)

# Load the regressor file into a DataFrame
regressor_df = pd.read_csv(f"{mdmr_dir}/regressor/{args.regressor_file}")
result_dir = MDMR_output_dir / f"{args_smoothness}mm" / str(group_num) / variable_name / "result"
cluster_report_filename = result_dir / "significant_cluster_report.csv"

cluster_report_df = pd.read_csv(cluster_report_filename)
df_numeric_cluster = cluster_report_df[cluster_report_df["Cluster ID"].apply(lambda x: str(x).isnumeric())][["Cluster ID", "Center of Mass AAL Label"]]

# Function to fetch Yeo network atlas based on version (7 thin, 7 thick, 17 thin, 17 thick)
def get_yeo_atlas(version):
    atlas = datasets.fetch_atlas_yeo_2011()
    if version == '7_thin':
        return atlas.thin_7
    elif version == '7_thick':
        return atlas.thick_7
    elif version == '17_thin':
        return atlas.thin_17
    elif version == '17_thick':
        return atlas.thick_17
    else:
        raise ValueError("Unknown version")

# Function to process each subject (needs to be at the top level for multiprocessing)
def process_subject_wrapper(args):
    subject_id, cluster_id, aal_label, version = args
    return process_subject(subject_id, cluster_id, aal_label, version)

# Function to process each subject for a specific Yeo network version
def process_subject(subject_id, cluster_id, aal_label, version):
    # Define file paths
    func_filename = f"{fmri_prep_dir}/sub-{subject_id}/ses-01/func/sub-{subject_id}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed{args_smoothness}mm_resampled4mm_naturebold.nii.gz"
    roi_filename = f"{MDMR_output_dir}/{args_smoothness}mm/{group_num}/{variable_name}/result/cluster_masks/MDMR_significant_aal({aal_label})_label({cluster_id}).nii.gz"
    brain_mask_filename = f"{mdmr_dir}/template/all_final_group_mask_{args_smoothness}mm.nii.gz"
    
    # Check if files exist
    if not os.path.exists(func_filename):
        return None  # Skip this subject
    if not os.path.exists(roi_filename):
        return None  # Skip this subject
    if not os.path.exists(brain_mask_filename):
        return None  # Skip this subject
    
    # Load and prepare the seed ROI mask
    seed_mask_img = load_img(roi_filename)
    brain_mask_img = load_img(brain_mask_filename)
    seed_mask_img_resampled = resample_to_img(seed_mask_img, brain_mask_img, interpolation='nearest')
    
    # Create a NiftiMasker for the seed ROI
    seed_masker = NiftiMasker(mask_img=seed_mask_img_resampled)
    
    # Extract the mean time series from the seed ROI
    seed_time_series = seed_masker.fit_transform(func_filename)
    mean_seed_time_series = np.mean(seed_time_series, axis=1)
    
    # Fetch the appropriate Yeo atlas for the given version
    atlas_filename = get_yeo_atlas(version)
    
    # Create a NiftiLabelsMasker for the Yeo atlas
    labels_masker = NiftiLabelsMasker(
        labels_img=atlas_filename,
        standardize=True,
        mask_img=brain_mask_img
    )

    # Extract time series for each network in the Yeo atlas
    labels_time_series = labels_masker.fit_transform(func_filename)

    # Get the labels used
    labels = labels_masker.labels_

    # Exclude 'Background' label (assumed to be label 0)
    label_indices = [i for i, label in enumerate(labels) if label != 0]

    # Mapping from label numbers to network names
    if '17' in version:
        # 17-network mapping
        yeo_17_network_names = {
            1: "Visual Central (Visual A)",
            2: "Visual Peripheral (Visual B)",
            3: "Somatomotor A",
            4: "Somatomotor B",
            5: "Dorsal Attention A",
            6: "Dorsal Attention B",
            7: "Salience/Ventral Attention A",
            8: "Salience/Ventral Attention B",
            9: "Limbic A",
            10: "Limbic B",
            11: "Control C",
            12: "Control A",
            13: "Control B",
            14: "Temporal Parietal",
            15: "Default Mode C",
            16: "Default Mode A",
            17: "Default Mode B"
        }
        network_labels_dynamic = [yeo_17_network_names.get(int(label), f'Network_{int(label)}') for label in labels if label != 0]
    else:
        # 7-network mapping
        label_to_network = {
            1: 'Visual',
            2: 'Somatomotor',
            3: 'Dorsal Attention',
            4: 'Ventral Attention',
            5: 'Limbic',
            6: 'Frontoparietal',
            7: 'Default'
        }
        network_labels_dynamic = [label_to_network.get(int(label), f'Network_{int(label)}') for label in labels if label != 0]

    # Compute correlations with the seed time series
    correlations = [
        pearsonr(mean_seed_time_series, labels_time_series[:, i])[0]
        for i in label_indices
    ]

    # Create a DataFrame for the current participant
    participant_df = pd.DataFrame({
        'Participant': [subject_id] * len(network_labels_dynamic),
        'Network': network_labels_dynamic,
        'Correlation': correlations,
        'Cluster_ID': [cluster_id] * len(network_labels_dynamic),
        'AAL_Label': [aal_label] * len(network_labels_dynamic),
        'Version': [version] * len(network_labels_dynamic)
    })

    return participant_df

# Wrapper function to process all subjects for a given cluster and version
def process_cluster(cluster_id, aal_label, version):
    print(f"Processing Cluster ID = {cluster_id}, AAL Label = {aal_label}, Version = {version}")
    
    results_list = []
    
    # Prepare argument list for each subject
    args_list = [(subject_id, cluster_id, aal_label, version) for subject_id in participant_list]
    
    # Use multiprocessing to process subjects in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        with tqdm(total=len(participant_list), desc=f"Cluster {cluster_id}, Version {version}") as pbar:
            for result in pool.imap_unordered(process_subject_wrapper, args_list):
                if result is not None:
                    results_list.append(result)
                pbar.update()

    if results_list:
        return pd.concat(results_list, ignore_index=True)
    else:
        return pd.DataFrame()

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame()

# Define Yeo network versions to analyze
versions = ['7_thin', '7_thick', '17_thin', '17_thick']

# Process each cluster and save results for each Yeo network version
for version in versions:
    all_version_results = pd.DataFrame()
    for index, row in df_numeric_cluster.iterrows():
        cluster_id = row['Cluster ID']
        aal_label = row['Center of Mass AAL Label']
        
        cluster_results = process_cluster(cluster_id, aal_label, version)
        all_version_results = pd.concat([all_version_results, cluster_results], ignore_index=True)
    
    # Save results to a separate file for each version
    output_csv = f"{MDMR_output_dir}/{args_smoothness}mm/{group_num}/{variable_name}/functional_connectivity_{version}.csv"
    all_version_results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
