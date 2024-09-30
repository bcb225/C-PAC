import pandas as pd
import numpy as np
import argparse
import os
from nilearn.maskers import NiftiMasker
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel

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

regressor_df = pd.read_csv(
    f"{mdmr_dir}/regressor/{args.regressor_file}"
)

subjects_label = regressor_df["Participant"].values
# 필요한 열만 선택하여 디자인 매트릭스 생성

extra_info_subjects = pd.DataFrame({
    "subject_label": subjects_label,
    variable_name: regressor_df[variable_name],
    "sex": regressor_df["SEX"],
    "age": regressor_df["AGE"],
    "yr_edu": regressor_df["YR_EDU"],
    "mean_framewise_displacement": regressor_df["Mean_Framewise_Displacement"]
})

design_matrix = make_second_level_design_matrix(
    subjects_label, extra_info_subjects
)
z_map_parent_path = seed_anal_dir / f"{args.smoothness}mm" / "corr_z-map"/ str(group_num) / variable_name
roi_list = [d.name for d in z_map_parent_path.iterdir() if d.is_dir()]

for roi in roi_list:
    print(roi)

"""z_maps = [f"{seed_anal_dir}///sub-{subject_id}_fisher_z_img.nii.gz" for subject_id in regressor_df['Participant']]
second_level_model = SecondLevelModel(n_jobs=-1)
second_level_model = second_level_model.fit(
    z_maps,
    design_matrix=design_matrix,
)"""