import pandas as pd
import argparse
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting
import os

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
# 필요한 열만 선택하여 디자인 매트릭스 생성
design_matrix = regressor_df[['SEX', 'AGE', 'YR_EDU', args.variable, 'Mean_Framewise_Displacement']]

# 인터셉트 열 추가
design_matrix.insert(0, 'intercept', 1)

z_maps = [f"{seed_anal_dir}/{args.smoothness}mm/corr_z-map/seed_{args.group}_{args.variable}/sub-{subject_id}_fisher_z_img.nii.gz" for subject_id in regressor_df['Participant']]

second_level_model = SecondLevelModel(n_jobs=-1)
second_level_model.fit(z_maps, design_matrix=design_matrix)

p_map = second_level_model.compute_contrast(args.variable, output_type='p_value')

result_dir = f"{seed_anal_dir}/{args.smoothness}mm/p_map/{args.group}/{args.variable}/result/"
temp_dir = f"{seed_anal_dir}/{args.smoothness}mm/p_map/{args.group}/{args.variable}/temp/"
os.makedirs(result_dir, exist_ok=True)

p_map_filename = f"{result_dir}/pvalue_map.nii.gz"
p_map.to_filename(p_map_filename)