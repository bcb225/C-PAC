import os
import pandas as pd
import nibabel as nib
import numpy as np
from nilearn.image import math_img, mean_img, resample_to_img
from nilearn.image import load_img, new_img_like
from tqdm import tqdm
import argparse
from nilearn.image import resample_img
import multiprocessing as mp

# Function to process each subject's mask
def process_subject_mask(func, maskdir, index):
    # Load functional data using nilearn
    func_img = load_img(func)
    func_data = func_img.get_fdata()

    # Create mask: voxels that are non-zero at any timepoint AND have non-zero variance
    subject_mask_data = np.logical_and(
        np.any(func_data != 0, axis=-1),  # Voxel has non-zero values at any timepoint
        np.std(func_data, axis=-1) != 0   # Voxel has non-zero variance across timepoints
    ).astype(np.uint8)

    # Create a new Nifti image using nilearn
    subject_mask_img = new_img_like(func_img, subject_mask_data)

    # Save the mask
    mask_img_path = os.path.join(maskdir, f"mask{index}.nii.gz")
    subject_mask_img.to_filename(mask_img_path)

    return mask_img_path

def process_fmri(subject_group, smoothness):
    # File paths
    input_file = f"../../regressor/{subject_group}_code_list.csv"
    base_path = "/mnt/NAS2-2/data/SAD_gangnam_resting_2/fMRIPrep_total"
    mask_suffix = "ses-01/func"
    maskdir = "../../template/"
    maskfile = f"{maskdir}/{subject_group}_group_mask_{smoothness}mm.nii.gz"
    grey_matter = "../../template/tpl-MNI152NLin2009cAsym_space-MNI_res-01_class-GM_probtissue.nii.gz"
    resampled_gm_file = f"{maskdir}/{subject_group}_resampled_gm_{smoothness}mm.nii.gz"
    threshold_gm_file = f"{maskdir}/{subject_group}_threshold_gm_{smoothness}mm.nii.gz"
    final_mask_file = f"{maskdir}/{subject_group}_final_group_mask_{smoothness}mm.nii.gz"
    group_prop_mask = f"{maskdir}/{subject_group}_group_prop_subjs_{smoothness}mm.nii.gz"

    # Read subject codes
    df = pd.read_csv(input_file, header=None)
    funcpaths = []

    for subject_code in df[0]:
        func_path = os.path.join(base_path, f"sub-{subject_code}/{mask_suffix}/sub-{subject_code}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed{smoothness}mm_resampled4mm_scrbold.nii.gz")
        if os.path.exists(func_path):
            funcpaths.append(func_path)
        else:
            print(f"File not found: {func_path}")

    if len(funcpaths) == 0:
        print("No valid fMRI files found.")
        return

    # Process subject masks in parallel
    print("Processing subject masks in parallel...")
    with mp.Pool(mp.cpu_count()) as pool:
        mask_imgs = list(tqdm(pool.starmap(process_subject_mask, [(func, maskdir, i) for i, func in enumerate(funcpaths)]),
                              total=len(funcpaths)))

    # Combine masks across subjects
    print("Calculating mean mask...")
    mean_mask = mean_img(mask_imgs)
    mean_mask.to_filename(group_prop_mask)

    # Debug: Print min and max values of mean_mask
    mean_mask_data = mean_mask.get_fdata()
    print("Mean mask min value:", mean_mask_data.min())
    print("Mean mask max value:", mean_mask_data.max())

    # Adjust threshold here (all subjects should have the voxel valid)
    threshold = 1
    final_mask = math_img(f"img >= {threshold}", img=mean_mask)
    final_mask.to_filename(maskfile)

    # Check number of active voxels in final_mask
    final_mask_data = final_mask.get_fdata()
    active_voxels_final_mask = np.count_nonzero(final_mask_data)
    print(f"Number of active voxels in final mask before combining with GM mask: {active_voxels_final_mask}")

    # Resample GM file to 4x4x4mm resolution explicitly
    print("Resampling GM file to 4x4x4mm resolution...")
    grey_matter_img = nib.load(grey_matter)
    target_affine = np.diag((4, 4, 4))  # Explicitly set the affine for 4x4x4mm
    resampled_gm = resample_img(grey_matter_img, target_affine=target_affine)
    #resampled_gm = resample_to_img(binary_gm, final_mask, interpolation="nearest")
    resampled_gm.to_filename(resampled_gm_file)

    # Debug: Print min and max values of resampled_gm
    resampled_gm_data = resampled_gm.get_fdata()
    print("Resampled GM min value:", resampled_gm_data.min())
    print("Resampled GM max value:", resampled_gm_data.max())

    # Applying threshold to GM file to create binary mask
    print("Applying threshold to GM file (0.25) to create binary GM mask...")
    thresholded_gm = math_img("img > 0.25", img=resampled_gm)
    thresholded_gm_data = thresholded_gm.get_fdata().astype(np.uint8)  # Convert to binary
    binary_gm = new_img_like(thresholded_gm, thresholded_gm_data)
    binary_gm.to_filename(threshold_gm_file)

    # Check number of active voxels in binary GM mask
    active_voxels_thresholded_gm = np.count_nonzero(thresholded_gm_data)
    print(f"Number of active voxels in binary GM mask: {active_voxels_thresholded_gm}")
    print("Resampling binary GM mask to match the final group mask...")
    resampled_binary_gm = resample_to_img(binary_gm, final_mask, interpolation="nearest")
    # Combining final group mask and binary GM mask
    print("Combining final group mask and binary GM mask...")
    combined_mask = math_img("img1 * img2", img1=final_mask, img2=resampled_binary_gm)
    combined_mask_data = combined_mask.get_fdata().astype(np.uint8)  # Ensure binary mask
    binary_combined_mask = new_img_like(combined_mask, combined_mask_data)
    binary_combined_mask.to_filename(final_mask_file)

    # Final count of active voxels
    final_img = nib.load(final_mask_file)
    voxel_count = np.count_nonzero(final_img.get_fdata())
    print(f"Number of active voxels in final mask after combining with GM mask: {voxel_count}")

    # Delete temporary mask files
    print("Deleting temporary mask files...")
    for i in tqdm(range(len(funcpaths)), desc="Deleting mask files"):
        mask_file = f"{maskdir}/mask{i}.nii.gz"
        if os.path.exists(mask_file):
            os.remove(mask_file)

if __name__ == "__main__":
    # Call the function with desired parameters
    process_fmri("all", 8)
