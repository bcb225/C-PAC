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
from itertools import repeat
import os
from tqdm import tqdm
import pandas as pd
import warnings
import argparse

# Ignore specific FutureWarning related to nilearn
warnings.filterwarnings("ignore", category=FutureWarning, module="nilearn")

def process_single_subject(subject_dir, scrub):
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

            # Apply 6mm smoothing to the original image
            fmri_smoothed_img = image.smooth_img(fmri_img, fwhm=8)

            # Resample the smoothed image to 4x4x4 mm voxel size
            fmri_resampled_img = resample_img(fmri_smoothed_img, target_affine=np.diag((4, 4, 4)))

            # Load the confounds
            confounds_df = pd.read_csv(confound_filename, sep='\t')

            # Calculate mean framewise displacement (FD)
            mean_fd = confounds_df['framewise_displacement'].mean()

            # Calculate the number of volumes where std_dvars > 1.5 or FD > 0.5mm
            num_volumes = len(confounds_df)
            #num_high_dvars_or_fd = ((confounds_df['std_dvars'] > 1.5) | (confounds_df['framewise_displacement'] > 0.5)).sum()
            num_high_fd = ((confounds_df['framewise_displacement'] > 0.5)).sum()
            percentage_high_dvars_or_fd = (num_high_fd / num_volumes) * 100

            # Load confounds with desired strategy
            confounds, sample_mask = load_confounds(
                img_files=str(fmri_filename),
                strategy=('motion', 'high_pass', 'wm_csf', 'global_signal'),
                motion='full',
                wm_csf='basic',
                global_signal='basic',
                demean=True
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
            output_filename = f"{subject_dir.name}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed8mm_resampled4mm_{scrub}bold.nii.gz"
            output_path = func_dir / output_filename

            # Save the denoised image
            fmri_denoised_img.to_filename(str(output_path))

            # Calculate the timeseries lengths
            resampled_length = fmri_resampled_img.shape[-1]
            denoised_length = fmri_denoised.shape[0] + 4  # Adding the discarded volumes

            # Return relevant information for further analysis
            return (resampled_length, denoised_length, mean_fd, percentage_high_dvars_or_fd, subject_dir.name)

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

def process_fmri_data_in_parallel_and_save(root_dir, output_csv_path, scrub):
    root_path = Path(root_dir)

    # Get all the subject directories
    subject_dirs = [subject_dir for subject_dir in root_path.glob("sub-*") if subject_dir.is_dir()]

    # To store the results for each subject
    results = []

    # Use tqdm to track the progress
    with ProcessPoolExecutor() as executor:
        # Wrap the map with tqdm for progress tracking
        for result in tqdm(executor.map(process_single_subject, subject_dirs, repeat(scrub)),
                           total=len(subject_dirs), desc="Processing fMRI Data"):
            if result is not None:
                results.append(result)

    total_subjects = len(results)
    if total_subjects == 0:
        print("No valid subjects found.")
        return

    # Initialize sets to track participants meeting each criterion
    participants_c1 = set()  # Criterion 1: mean FD > 2mm
    participants_c2 = set()  # Criterion 2: >20% volumes with std_dvars > 1.5 or FD > 0.5mm

    # Loop over results to populate the sets
    valid_participant_codes = []
    for r in results:
        resampled_length, denoised_length, mean_fd, percentage_high_dvars_or_fd, participant_code = r

        c1 = mean_fd > 2
        c2 = percentage_high_dvars_or_fd > 20

        if c1:
            participants_c1.add(participant_code)
        if c2:
            participants_c2.add(participant_code)

        # Exclude participants meeting any criteria
        if not (c1 or c2):
            if denoised_length >= 0.8 * resampled_length:
                valid_participant_codes.append(participant_code)

    # Calculate counts for each criterion
    count_c1 = len(participants_c1)
    count_c2 = len(participants_c2)

    # Calculate counts for both criteria
    participants_c1_c2 = participants_c1 & participants_c2
    count_c1_c2 = len(participants_c1_c2)

    # Print the counts and proportions
    print(f"Total subjects processed: {total_subjects}")
    print(f"Subjects with mean FD > 2mm (Criterion 1): {count_c1} ({(count_c1/total_subjects)*100:.2f}%)")
    print(f"Subjects with >20% volumes with FD > 0.5mm (Criterion 2): {count_c2} ({(count_c2/total_subjects)*100:.2f}%)")
    print(f"Subjects meeting both Criteria 1 and 2: {count_c1_c2}")

    # Save valid participant codes to CSV file
    valid_df = pd.DataFrame({"Participant": valid_participant_codes})
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    valid_df.to_csv(output_path, index=False)

    print(f"Valid participant codes saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process fMRI data and save valid participant codes.")
    parser.add_argument('--scrub', required=True, help="Naming of the file.")

    args = parser.parse_args()
    process_fmri_data_in_parallel_and_save(
        root_dir="/mnt/NAS2-2/data/SAD_gangnam_resting_2/fMRIPrep_total/",
        output_csv_path="/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/input/valid_after_post_fmriprep_processing.csv",
        scrub=args.scrub
    )
