from pathlib import Path
import nibabel as nib
from nilearn import image
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import NiftiMasker
import json
from nilearn.interfaces.fmriprep import load_confounds_strategy
import numpy as np
from nilearn.image import resample_img
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm
import pandas as pd
import warnings

# Ignore specific FutureWarning related to nilearn
warnings.filterwarnings("ignore", category=FutureWarning, module="nilearn")

def process_single_subject(subject_dir):
    try:
        func_dir = subject_dir / 'ses-01' / 'func'

        # Locate the fMRI and JSON files
        fmri_filename = func_dir / f'{subject_dir.name}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
        json_filename = func_dir / f'{subject_dir.name}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.json'
        confound_filename = func_dir / f'{subject_dir.name}_ses-01_task-rest_desc-confounds_timeseries.tsv'

        # Check if all required files exist
        if fmri_filename.exists() and json_filename.exists() and confound_filename.exists():
            # Load the JSON file
            with open(json_filename, 'r') as f:
                metadata = json.load(f)

            # Extract the TR
            tr = metadata.get('RepetitionTime', None)
            if tr is None:
                print(f"TR not found in JSON for {subject_dir.name}. Skipping.")
                return None

            # Load the fMRI data
            fmri_img = image.load_img(str(fmri_filename))

            # Apply 8mm smoothing to the original image
            fmri_smoothed_img = image.smooth_img(fmri_img, fwhm=8)

            # Resample the smoothed image to 4x4x4 mm voxel size
            fmri_resampled_img = resample_img(fmri_smoothed_img, target_affine=np.diag((4, 4, 4)))

            # Load the confounds using load_confounds
            confounds_df = pd.read_csv(confound_filename, sep='\t')

            # Calculate mean framewise displacement (FD)
            mean_fd = confounds_df['framewise_displacement'].mean()

            # Load confounds with desired strategy
            confounds, sample_mask = load_confounds_strategy(
                str(fmri_filename), 
                denoise_strategy="simple",  # Change to "scrubbing" if needed
                global_signal="basic",
                # Uncomment and set thresholds if using scrubbing
                # fd_threshold=0.5, 
                # std_dvars_threshold=1.5
            )

            # Discard the first four volumes
            if sample_mask is None:
                sample_mask = np.arange(fmri_resampled_img.shape[-1])  # Include all volumes
            sample_mask = sample_mask[sample_mask >= 4]  # Exclude the first four volumes

            # Create a NiftiMasker without additional smoothing
            masker = NiftiMasker(
                smoothing_fwhm=None,  # No additional smoothing since it's already applied
                standardize=True,
                detrend=False,
                high_pass=None,       # No additional high-pass filtering
                low_pass=0.1,         # Apply low-pass filter at 0.1 Hz
                t_r=tr,               # Use the TR extracted from the metadata
            )

            # Fit the masker (this computes the mask)
            masker.fit(fmri_resampled_img)

            # Apply the masker to perform confound regression and filtering
            fmri_denoised = masker.transform(
                fmri_resampled_img,
                confounds=confounds,
                sample_mask=sample_mask
            )

            # Inverse transform to get a denoised Nifti image
            fmri_denoised_img = masker.inverse_transform(fmri_denoised)

            # Define the output filename following the desired convention
            output_filename = f"{subject_dir.name}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed8mm_resampled4mm_scrbold.nii.gz"
            output_path = func_dir / output_filename

            # Save the denoised image
            fmri_denoised_img.to_filename(str(output_path))

            # Calculate the timeseries lengths
            resampled_length = fmri_resampled_img.shape[-1]
            denoised_length = fmri_denoised.shape[0] + 4  # Adding the discarded volumes

            # Return relevant information for further analysis
            return resampled_length, denoised_length, mean_fd, subject_dir.name

        else:
            missing = []
            if not fmri_filename.exists():
                missing.append(fmri_filename.name)
            if not json_filename.exists():
                missing.append(json_filename.name)
            if not confound_filename.exists():
                missing.append(confound_filename.name)
            print(f"Missing files for {subject_dir.name}: {', '.join(missing)}. Skipping.")
            return None
    except Exception as e:
        print(f"Error processing {subject_dir.name}: {e}")
        return None

def process_fmri_data_in_parallel_and_save(root_dir, output_csv_path):
    root_path = Path(root_dir)

    # Get all the subject directories
    subject_dirs = [subject_dir for subject_dir in root_path.glob("sub-*") if subject_dir.is_dir()]

    # To store the results for each subject
    results = []

    # Use tqdm to track the progress
    with ProcessPoolExecutor() as executor:
        # Wrap the map with tqdm for progress tracking
        for result in tqdm(executor.map(process_single_subject, subject_dirs), total=len(subject_dirs), desc="Processing fMRI Data"):
            if result is not None:
                results.append(result)

    # Calculate the number of subjects with different criteria
    total_subjects = len(results)
    if total_subjects == 0:
        print("No valid subjects found.")
        return

    # Calculate the proportions and counts for each criterion
    count_r2_gt_2 = sum(1 for r in results if r[2] > 2)
    count_r1_lt_08_r0 = sum(1 for r in results if r[1] < 0.8 * r[0])
    count_valid_subjects = sum(1 for r in results if r[2] <= 2 and r[1] >= 0.8 * r[0])

    # Calculate the proportions
    proportion_r2_gt_2 = count_r2_gt_2 / total_subjects
    proportion_r1_lt_08_r0 = count_r1_lt_08_r0 / total_subjects
    proportion_valid_subjects = count_valid_subjects / total_subjects

    # Print the counts and proportions
    print(f"Subjects with mean FD > 2: {count_r2_gt_2} ({proportion_r2_gt_2:.2%})")
    print(f"Subjects with denoised_length < 0.8 * resampled_length: {count_r1_lt_08_r0} ({proportion_r1_lt_08_r0:.2%})")
    print(f"Valid subjects (mean FD <= 2 and denoised_length >= 0.8 * resampled_length): {count_valid_subjects} ({proportion_valid_subjects:.2%})")

    # Extract participant codes from valid subjects
    # To do this, we need to match the results with subject_dirs
    # Since results are appended in the same order as executor.map, we can zip them accordingly
    valid_participant_codes = [r[3] for r in results if r[2] <= 2 and r[1] >= 0.8 * r[0]]

    # Save to CSV file
    valid_df = pd.DataFrame({"Participant": valid_participant_codes})
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    valid_df.to_csv(output_path, index=False)

    print(f"Valid participant codes saved to {output_csv_path}")

# Example usage:
process_fmri_data_in_parallel_and_save(
    root_dir="/mnt/NAS2-2/data/SAD_gangnam_resting_2/fMRIPrep_total/",
    output_csv_path="/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/input/valid_after_post_fmriprep_processing.csv"
)
