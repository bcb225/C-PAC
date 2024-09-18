import numpy as np
from sklearn.utils import shuffle
from nilearn.glm.second_level import SecondLevelModel, make_second_level_design_matrix
from nilearn import image
import nibabel as nib
import os
from tqdm import tqdm
import pandas as pd
def permutation_test(model, z_maps, design_matrix, contrast_def, output_dir, mask_img, n_permutations=15000):
    """Perform permutation test and calculate individual p-value maps for each permutation."""
    
    # Original contrast calculation
    original_contrast = model.compute_contrast(
        second_level_contrast=contrast_def,
        output_type='stat'
    )
    original_stat_map = np.abs(original_contrast.get_fdata())
    
    # Save original statistical map
    original_stat_img = image.new_img_like(mask_img, original_stat_map)
    nib.save(original_stat_img, os.path.join(output_dir, 'original_stat_map.nii.gz'))
    
    # Permutation
    perm_stat_maps = []
    os.makedirs(os.path.join(output_dir, 'permutations'), exist_ok=True)
    
    for _ in tqdm(range(n_permutations), desc="Performing permutations"):
        # Shuffle only the variable of interest (contrast)
        shuffled_data = design_matrix.copy()
        shuffled_data[contrast_def] = shuffle(shuffled_data[contrast_def].values)
        
        # Refit the model
        perm_model = SecondLevelModel(mask_img=mask_img, n_jobs=-1)
        perm_model.fit(z_maps, design_matrix=shuffled_data)
        
        # Calculate contrast with shuffled data
        perm_contrast = perm_model.compute_contrast(
            second_level_contrast=contrast_def,
            output_type='stat'
        )
        perm_stat_map = np.abs(perm_contrast.get_fdata())
        perm_stat_maps.append(perm_stat_map)
    
    perm_stat_maps = np.array(perm_stat_maps)
    
    # Calculate and save individual p-value maps for each permutation
    for i in tqdm(range(n_permutations), desc="Calculating p-value maps"):
        nth_p_value_map = np.mean(perm_stat_maps >= perm_stat_maps[i], axis=0)
        p_value_img = image.new_img_like(mask_img, nth_p_value_map)
        nib.save(p_value_img, os.path.join(output_dir, 'permutations', f'p_value_map_{i+1}.nii.gz'))
    
    # Calculate and save overall p-value map
    overall_p_value_map = np.mean(perm_stat_maps >= original_stat_map, axis=0)
    overall_p_value_img = image.new_img_like(mask_img, overall_p_value_map)
    nib.save(overall_p_value_img, os.path.join(output_dir, 'overall_p_value_map.nii.gz'))
    
    print(f"All permutation results saved in {os.path.join(output_dir, 'permutations')}")
    print(f"Overall p-value map saved as {os.path.join(output_dir, 'overall_p_value_map.nii.gz')}")

# The rest of your code remains the same
group = "gangnam_sad"
variable = "LSAS"
smoothness = 6
mdmr_dir = os.path.expanduser("~/fmri_project/C-PAC/CPAC/bcb_mdmr/")
nas_dir = os.path.expanduser("~/fmri_project/C-PAC/CPAC/bcb_mdmr/output/")
MDMR_output_dir = f"{nas_dir}/SAD_gangnam_MDMR/"
fmri_prep_dir = f"{nas_dir}/SAD_gangnam_resting_2/fMRIPrep_total"
seed_anal_dir = f"{nas_dir}/SAD_gangnam_seed_based_analysis/"

regressor_df = pd.read_csv(
    f"{mdmr_dir}/input/{group}_{variable}_regressor.csv"
)

subjects_label = regressor_df["Participant"].values
# Select only necessary columns to create design matrix
extra_info_subjects = pd.DataFrame({
    "subject_label": subjects_label,
    "LSAS": regressor_df["LSAS"],
    "sex": regressor_df["SEX"],
    "age": regressor_df["AGE"],
    "yr_edu": regressor_df["YR_EDU"],
    "mean_framewise_displacement": regressor_df["Mean_Framewise_Displacement"]
})
design_matrix = make_second_level_design_matrix(
    subjects_label, extra_info_subjects
)
z_maps = [f"{seed_anal_dir}/{smoothness}mm/corr_z-map/seed_{group}_{variable}/sub-{subject_id}_fisher_z_img.nii.gz" for subject_id in regressor_df['Participant']]
second_level_model = SecondLevelModel(n_jobs=-1)
second_level_model = second_level_model.fit(
    z_maps,
    design_matrix=design_matrix,
)

# Define contrast (example)
contrast_def = 'LSAS'  # or other appropriate contrast definition

# Load mask image
mask_img = image.load_img(f"{mdmr_dir}/template/gangnam_total_final_group_mask_{smoothness}mm.nii.gz")

# Create result storage directory
output_dir = './permutation_results'
os.makedirs(output_dir, exist_ok=True)

# Perform permutation test and save results
permutation_test(second_level_model, z_maps, design_matrix, contrast_def, output_dir, mask_img)