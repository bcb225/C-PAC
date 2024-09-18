import numpy as np
from nilearn import image
from nilearn.glm import second_level
import pandas as pd
import os
from joblib import Parallel, delayed
import argparse

def permute_and_analyze(z_maps, design_matrix, target_column, seed):
    np.random.seed(seed)
    permuted_design = design_matrix.copy()
    permuted_design[target_column] = np.random.permutation(permuted_design[target_column])
    
    second_level_model = second_level.SecondLevelModel()
    second_level_model.fit(z_maps, design_matrix=permuted_design)
    
    contrast = second_level_model.compute_contrast(target_column, output_type='z_score')
    
    return contrast

def generate_pvalue_map(permutation_maps, original_zmap):
    original_data = original_zmap.get_fdata()
    p_values = np.zeros_like(original_data)
    for i in range(original_data.shape[0]):
        for j in range(original_data.shape[1]):
            for k in range(original_data.shape[2]):
                original_z = original_data[i, j, k]
                perm_z_scores = [perm_map.get_fdata()[i, j, k] for perm_map in permutation_maps]
                p_values[i, j, k] = np.mean(np.abs(perm_z_scores) >= np.abs(original_z))
    
    p_value_map = image.new_img_like(original_zmap, p_values)
    return p_value_map

def generate_multiple_pvalue_maps(z_maps, design_matrix, target_column, n_permutations, batch_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    original_model = second_level.SecondLevelModel()
    original_model.fit(z_maps, design_matrix=design_matrix)
    original_zmap = original_model.compute_contrast(target_column, output_type='z_score')
    
    for batch_start in range(0, n_permutations, batch_size):
        batch_end = min(batch_start + batch_size, n_permutations)
        batch_size = batch_end - batch_start
        
        permutation_maps = Parallel(n_jobs=-1)(
            delayed(permute_and_analyze)(z_maps, design_matrix, target_column, i) 
            for i in range(batch_start, batch_end)
        )
        
        for i in range(batch_size):
            current_permutation_maps = permutation_maps[:i+1]
            p_value_map = generate_pvalue_map(current_permutation_maps, original_zmap)
            p_value_map.to_filename(os.path.join(output_dir, f"pvalue_map_{batch_start+i+1}.nii.gz"))
        
        print(f"Generated p-value maps {batch_start+1} to {batch_end}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run permutation test for second-level analysis.')
    parser.add_argument('--group', type=str, required=True, help='Subject group name')
    parser.add_argument('--variable', type=str, required=True, help='Variable of interest')
    parser.add_argument('--smoothness', type=int, required=True, help='Smoothness of preprocessing')
    parser.add_argument('--n_permutations', type=int, default=15000, help='Number of permutations')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of permutations per batch')
    args = parser.parse_args()

    mdmr_dir = os.path.expanduser("~/fmri_project/C-PAC/CPAC/bcb_mdmr/")
    nas_dir = os.path.expanduser("~/fmri_project/C-PAC/CPAC/bcb_mdmr/output/")
    seed_anal_dir = f"{nas_dir}/SAD_gangnam_seed_based_analysis/"
    
    regressor_df = pd.read_csv(f"{mdmr_dir}/input/{args.group}_{args.variable}_regressor.csv")
    subjects_label = regressor_df["Participant"].values
    
    extra_info_subjects = pd.DataFrame({
        "subject_label": subjects_label,
        args.variable: regressor_df[args.variable],
        "sex": regressor_df["SEX"],
        "age": regressor_df["AGE"],
        "yr_edu": regressor_df["YR_EDU"],
        "mean_framewise_displacement": regressor_df["Mean_Framewise_Displacement"]
    })
    
    design_matrix = second_level.make_second_level_design_matrix(subjects_label, extra_info_subjects)
    
    z_maps = [f"{seed_anal_dir}/{args.smoothness}mm/corr_z-map/seed_{args.group}_{args.variable}/sub-{subject_id}_fisher_z_img.nii.gz" for subject_id in subjects_label]
    
    output_dir = f"{seed_anal_dir}/{args.smoothness}mm/pvalue_map/seed_{args.group}_{args.variable}/"
    
    generate_multiple_pvalue_maps(z_maps, design_matrix, args.variable, args.n_permutations, args.batch_size, output_dir)

    print(f"All {args.n_permutations} p-value maps have been generated and saved in {output_dir}")