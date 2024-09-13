import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.second_level import make_second_level_design_matrix

from tqdm import tqdm
import os
import argparse
from multiprocessing import Pool, cpu_count

def run_permutation_test(args):
    z_maps, design_matrix, variable, permutation_num, smoothness, group, extra_info_subjects = args
    temp_dir = f"{seed_anal_dir}/{smoothness}mm/p_map/{group}/{variable}/temp/volume/"
    os.makedirs(temp_dir, exist_ok=True)
    
    permuted_variable = np.random.permutation(design_matrix[variable])
    permuted_design_matrix = design_matrix.copy()
    permuted_design_matrix[variable] = permuted_variable
    
    second_level_model = SecondLevelModel(n_jobs=-1)
    second_level_model.fit(z_maps, design_matrix=permuted_design_matrix)
    p_map = second_level_model.compute_contrast(variable, output_type='p_value')
    
    p_map_filename = f"{temp_dir}/{permutation_num}_pvalue_map.nii.gz"
    p_map.to_filename(p_map_filename)

def run_batch_permutation_test(z_maps, design_matrix, variable, start_idx, end_idx, smoothness, group, extra_info_subjects):
    args_list = [(z_maps, design_matrix, variable, i, smoothness, group, extra_info_subjects) for i in range(start_idx, end_idx + 1)]
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(run_permutation_test, args_list), total=len(args_list)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run permutation test for second-level analysis.')
    parser.add_argument('--group', type=str, required=True, help='Subject group name')
    parser.add_argument('--variable', type=str, required=True, help='Variable of interest')
    parser.add_argument('--smoothness', type=str, required=True, help='Smoothness of preprocessing')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of permutations per batch')
    args = parser.parse_args()

    mdmr_dir = "/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/"
    nas_dir = "/mnt/NAS2-2/data/"
    seed_anal_dir = f"{nas_dir}/SAD_gangnam_seed_based_analysis/"

    regressor_df = pd.read_csv(f"{mdmr_dir}/input/{args.group}_{args.variable}_regressor.csv")
    subjects_label = regressor_df["Participant"].values
    # 필요한 열만 선택하여 디자인 매트릭스 생성

    extra_info_subjects = pd.DataFrame({
        "subject_label": subjects_label,
        "sex": regressor_df["SEX"],
        "age": regressor_df["AGE"],
        "yr_edu": regressor_df["YR_EDU"],
        "mean_framewise_displacement": regressor_df["Mean_Framewise_Displacement"]
    })
    design_matrix = make_second_level_design_matrix(
        subjects_label, extra_info_subjects
    )

    z_maps = [f"{seed_anal_dir}/{args.smoothness}mm/corr_z-map/seed_{args.group}_{args.variable}/sub-{subject_id}_fisher_z_img.nii.gz"
              for subject_id in regressor_df['Participant']]

    total_permutations = 15000
    num_batches = (total_permutations + args.batch_size - 1) // args.batch_size

    for batch in range(num_batches):
        start_idx = batch * args.batch_size + 1
        end_idx = min((batch + 1) * args.batch_size, total_permutations)
        print(f"Running batch {batch + 1}/{num_batches} (permutations {start_idx}-{end_idx})")
        run_batch_permutation_test(z_maps, design_matrix, args.variable, start_idx, end_idx, args.smoothness, args.group, extra_info_subjects)

    print("All permutations completed.")