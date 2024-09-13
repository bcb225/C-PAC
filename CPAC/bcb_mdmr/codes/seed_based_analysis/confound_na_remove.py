import pandas as pd
import numpy as np
import argparse
import os
parser = argparse.ArgumentParser(description='Run CWAS analysis.')
parser.add_argument('--group', type=str, required=True, help='Subject group name')
parser.add_argument('--variable', type=str, required=True, help='Variable of interest')
parser.add_argument('--smoothness', type=str, required=True, help='Smoothness of preprocessing')
args = parser.parse_args()

mdmr_dir = "/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/"
nas_dir = "/mnt/NAS2-2/data/"

regressor_df = pd.read_csv(
    f"{mdmr_dir}/input/{args.group}_{args.variable}_regressor.csv"
)

subject_id_list = regressor_df["Participant"].values
for subject_id in subject_id_list:
    confounds = pd.read_csv(
        f"{nas_dir}/SAD_gangnam_resting_2/fMRIPrep_total/sub-{subject_id}/ses-01/func/sub-{subject_id}_ses-01_task-rest_desc-confounds_timeseries.tsv", 
        sep='\t',  # TSV 파일이므로 구분자를 탭으로 명시
        na_values=['n/a', 'N/A', 'NA', ''],  # 'n/a' 등을 NaN으로 처리
        keep_default_na=True
    )
    modified_confound_dir = f"{nas_dir}/SAD_gangnam_seed_based_analysis/modified_confounds/"
    confounds_filled = confounds.fillna(confounds.mean())
    os.makedirs(modified_confound_dir, exist_ok=True)
    confounds_filled.to_csv(f"{modified_confound_dir}/sub-{subject_id}_ses-01_task-rest_desc-confounds_timeseries_modified.tsv", sep='\t', index=False)