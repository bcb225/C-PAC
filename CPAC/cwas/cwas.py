import os
import time
from pathlib import Path
import nibabel as nb
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import t
from numpy import inf

from CPAC.cwas.mdmr import mdmr
from CPAC.utils import correlation

from CPAC.pipeline.cpac_ga_model_generator import (create_merge_mask,
                                                   create_merged_copefile)

from tqdm import tqdm
def joint_mask(subjects, mask_file=None):
    """
    Creates a joint mask (intersection) common to all the subjects in a provided list
    and a provided mask
    
    Parameters
    ----------
    subjects : dict of strings
        A length `N` list of file paths of the nifti files of subjects
    mask_file : string
        Path to a mask file in nifti format
    
    Returns
    -------
    joint_mask : string
        Path to joint mask file in nifti format
    
    """
    if not mask_file:
        files = list(subjects.values())
        cope_file = os.path.join(os.getcwd(), 'joint_cope.nii.gz')
        mask_file = os.path.join(os.getcwd(), 'joint_mask.nii.gz')
        create_merged_copefile(files, cope_file)
        create_merge_mask(cope_file, mask_file)
    return mask_file


def calc_mdmrs(D, regressor, cols, permutations, mask_file, base_dir, mode):
    cols = np.array(cols, dtype=np.int32)
    F_set, p_set = mdmr(D, regressor, cols, permutations, mask_file, base_dir, mode)
    return F_set, p_set


def calc_subdists(subjects_data, voxel_range, subject_group,target_subject_index,smoothness, subjects):
    distance_dir = Path(f"/mnt/NAS2-2/data/SAD_gangnam_MDMR/distance/{smoothness}mm/")
    distance_dir.mkdir(parents=True, exist_ok=True)
    distance_file_name = distance_dir / f"{subject_group}_distance.npy"
   
    if os.path.exists(distance_file_name):
        # 파일이 존재하는 경우: 파일을 로드
        print(f"Loading distance data from {distance_file_name}")
        distances = np.load(distance_file_name, mmap_mode='r')
        
        # 필요한 부분만 슬라이싱하여 로드
        distances = distances[voxel_range, :, :]
        print(distances.shape)
        if target_subject_index:
            print("slicing")
            distances = distances [:,target_subject_index,:][:,:,target_subject_index]
            # 이후 처리를 여기에 추가할 수 있습니다.
            # 예를 들어, distances를 이용한 분석이나 결과 반환
            #print(distances.shape)
            return distances
        else:
            #print(distances.shape)
            return distances
    else:
        print("Total participant and not calculated before.")
        print("Total participant distance should be calculated at least once")
        subjects, voxels, _ = subjects_data.shape
        D = np.zeros((len(voxel_range), subjects, subjects))
        for i, v in tqdm(enumerate(voxel_range), total=len(voxel_range)):
            profiles = np.zeros((subjects, voxels))
            for si in range(subjects):
                profiles[si] = correlation(subjects_data[si, v], subjects_data[si])
            profiles = np.clip(np.nan_to_num(profiles), -0.9999, 0.9999)
            profiles = np.arctanh(np.delete(profiles, v, 1))
            D[i] = correlation(profiles, profiles)

        D = np.sqrt(2.0 * (1.0 - D))
        np.save(distance_file_name, D)
        return D
        """else:
        print(f"Subset of participant and not calculated before.")
        print("The distance file can be extracted from the total distance file.")
        total_distance_filename = distance_dir / "1_distance.npy"
        total_distance = np.load(total_distance_filename)
        total_code_list = pd.read_csv("../../regressor/all_code_list.csv", header=None)
        curr_code_list = pd.Series(list(subjects.keys()))
        mapping = curr_code_list.apply(lambda x: total_code_list[total_code_list[0] == x].index.tolist())

        mapping = mapping.explode().reset_index()
        mapping.columns = ['curr_code_list', 'total_code_index']
        total_indices = mapping["total_code_index"].astype(int).values
        curr_distance = total_distance[:, total_indices][:, :, total_indices].astype(np.float64).copy()
        np.save(distance_file_name, curr_distance)
        return curr_distance
        """

def calc_cwas(subjects_data, regressor, regressor_selected_cols, permutations, voxel_range, subject_group,target_subject_index,smoothness, mask_file, base_dir, mode, subjects):
    start_time = time.time()
    D = calc_subdists(subjects_data, voxel_range, subject_group,target_subject_index,smoothness, subjects)
    end_time = time.time()

    elapsed_time = end_time - start_time
    #print(f"calc_subdists time: {elapsed_time:.2f} seconds")

    start_time = time.time()
    F_set, p_set = calc_mdmrs(
        D, regressor, regressor_selected_cols, permutations, mask_file, base_dir, mode
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"calc_mdmrs time: {elapsed_time:.2f} seconds")

    return F_set, p_set
def pval_to_zval(p_set, permu):
    inv_pval = 1 - p_set
    zvals = t.ppf(inv_pval, (len(p_set) - 1))
    zvals[zvals == -inf] = permu / (permu + 1)
    zvals[zvals == inf] = permu / (permu + 1)
    return zvals

def nifti_cwas(subjects, mask_file, regressor_file, participant_column,
               columns_string, permutations, variable_of_interest, smoothness, voxel_range, subject_group, target_subject_index, mode):
    """
    Performs CWAS for a group of subjects
    
    Parameters
    ----------
    subjects : dict of strings:strings
        A length `N` dict of id and file paths of the nifti files of subjects
    mask_file : string
        Path to a mask file in nifti format
    regressor_file : string
        file path to regressor CSV or TSV file (phenotypic info)
    columns_string : string
        comma-separated string of regressor labels
    permutations : integer
        Number of pseudo f values to sample using a random permutation test
    voxel_range : ndarray
        Indexes from range of voxels (inside the mask) to perform cwas on.
        Index ordering is based on the np.where(mask) command
    
    Returns
    -------
    F_file : string
        .npy file of pseudo-F statistic calculated for every voxel
    p_file : string
        .npy file of significance probabilities of pseudo-F values
    voxel_range : tuple
        Passed on by the voxel_range provided in parameters, used to make parallelization
        easier
        
    """
    base_dir = Path(f"/mnt/NAS2-2/data/SAD_gangnam_MDMR/{smoothness}mm/{subject_group}/{variable_of_interest}/result/")
    try:
        regressor_data = pd.read_table(regressor_file,
                                       sep=None, engine="python",
                                       dtype={ participant_column: str })
    except:
        regressor_data = pd.read_table(regressor_file,
                                       sep=None, engine="python")
        regressor_data = regressor_data.astype({ participant_column: str })

    # drop duplicates
    regressor_data = regressor_data.drop_duplicates()

    regressor_cols = list(regressor_data.columns)
    if not participant_column in regressor_cols:
        raise ValueError('Participant column was not found in regressor file.')

    if participant_column in columns_string:
        raise ValueError('Participant column can not be a regressor.')

    subject_ids = list(subjects.keys())
    subject_files = list(subjects.values())

    # Validate and filter subjects based on target_subject_index
    if target_subject_index:
        if any(idx >= len(subject_ids) for idx in target_subject_index):
            #print(f"Total Number of Participants: {len(subject_ids)}")
            #print(target_subject_index)
            raise ValueError('Index out of range in target_subject_index.')

        filtered_subject_ids = [subject_ids[idx] for idx in target_subject_index]
        filtered_subject_files = [subject_files[idx] for idx in target_subject_index]

        # Create a filtered subjects dictionary
        filtered_subjects = dict(zip(filtered_subject_ids, filtered_subject_files))
        subject_ids = list(filtered_subjects.keys())
        subject_files = list(filtered_subjects.values())
        
        # Filter regressor data based on the filtered_subject_ids
        filtered_regressor_data = regressor_data[regressor_data[participant_column].isin(filtered_subject_ids)]
    else:
        filtered_subjects = subjects
        subject_ids = list(filtered_subjects.keys())
        subject_files = list(filtered_subjects.values())
        filtered_regressor_data = regressor_data

    # check for inconsistency with leading zeroes
    # (sometimes, the sub_ids from individual will be something like
    #  '0002601' and the phenotype will have '2601')
    for index, row in filtered_regressor_data.iterrows():
        pheno_sub_id = str(row[participant_column])
        for sub_id in subject_ids:
            if str(sub_id).lstrip('0') == str(pheno_sub_id):
                filtered_regressor_data.at[index, participant_column] = str(sub_id)
    
    filtered_regressor_data.index = filtered_regressor_data[participant_column]

    # Keep only data from specific subjects
    ordered_regressor_data = filtered_regressor_data.loc[subject_ids]

    columns = columns_string.split(',')
    regressor_selected_cols = [
        i for i, c in enumerate(regressor_cols) if c in columns
    ]

    if len(regressor_selected_cols) == 0:
        regressor_selected_cols = [i for i, c in enumerate(regressor_cols)]
    regressor_selected_cols = np.array(regressor_selected_cols)
    # Remove participant id column from the dataframe and convert it to a numpy matrix
    regressor = ordered_regressor_data \
        .drop(columns=[participant_column]) \
        .reset_index(drop=True) \
        .values \
        .astype(np.float64)
    #print(f"Regressor shape: {regressor.shape}")
    print(f"Total Number of Participants: {regressor.shape[0]}")
    print(f"Total Number of Variable (including covariates): {regressor.shape[1]}")
    if len(regressor.shape) == 1:
        regressor = regressor[:, np.newaxis]
    elif len(regressor.shape) != 2:
        raise ValueError('Bad regressor shape: %s' % str(regressor.shape))
    if len(subject_files) != regressor.shape[0]:
        raise ValueError('Number of subjects does not match regressor size')
    mask = nb.load(mask_file).get_fdata().astype('bool')
    mask_indices = np.where(mask)
    timepoints_list = []
    for subject_file in subject_files:
        img = nb.load(subject_file)
        timepoints_list.append(img.shape[-1])
    min_timepoints = min(timepoints_list)
    print(f"Minimum number of timepoints across subjects: {min_timepoints}")
    subjects_data = np.array([
        nb.load(subject_file).get_fdata().astype('float64')[mask_indices][:, :min_timepoints]
        for subject_file in subject_files
    ])
    """subjects_data = np.array([
        nb.load(subject_file).get_fdata().astype('float64')[mask_indices]
        for subject_file in subject_files
    ])"""

    F_set, p_set = calc_cwas(subjects_data, regressor, regressor_selected_cols,
                             permutations, voxel_range, subject_group,target_subject_index,smoothness, mask_file, base_dir, mode, subjects)
    
    
    raw_dir = Path(f"/mnt/NAS2-2/data/SAD_gangnam_MDMR/{smoothness}mm/{subject_group}/{variable_of_interest}/temp/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    F_file = raw_dir /f"pseudo_F.npy"
    np.save(F_file, F_set)
    for i in range (0,p_set.shape[0]):
        p_file = raw_dir / f"significance_{i}.npy"
        np.save(p_file, p_set[i,:])
    temp_dir = Path(f"/mnt/NAS2-2/data/SAD_gangnam_MDMR/{smoothness}mm/{subject_group}/{variable_of_interest}/temp")
    # Ensure the directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir, voxel_range


def create_cwas_batches(mask_file, batches):
    mask = nb.load(mask_file).get_fdata().astype('bool')
    voxels = mask.sum(dtype=int)
    return np.array_split(np.arange(voxels), batches)


def volumize(mask_image, data):
    mask_data = mask_image.get_fdata().astype('bool')
    volume = np.zeros_like(mask_data, dtype=data.dtype)
    volume[np.where(mask_data == True)] = data
    return nb.Nifti1Image(
        volume,
        header=mask_image.header,
        affine=mask_image.affine
    )


def merge_cwas_batches(cwas_batches, mask_file, z_score, permutations, subject_group, variable_of_interest, smoothness):
    _, voxel_range = zip(*cwas_batches)
    voxels = np.array(np.concatenate(voxel_range))

    mask_image = nb.load(mask_file)

    F_set = np.zeros_like(voxels, dtype=np.float64)
    p_set = np.zeros_like(voxels, dtype=np.float64)
    p_set_dict = {}
    for i in range(0, permutations):
        p_set_dict[i] = np.zeros_like(voxels, dtype=np.float64)
    for temp_dir, voxel_range in cwas_batches:
        temp_dir = Path(temp_dir)
        F_file = temp_dir / "raw" / "pseudo_F.npy"
        F_set[voxel_range] = np.load(F_file)
        
        for i in range(0,permutations):
            p_file = temp_dir/ "raw" / f"significance_{i}.npy"
            p_set_dict[i][voxel_range] = np.load(p_file)
        

    log_p_set = -np.log10(p_set)
    one_p_set = 1 - p_set

    F_vol = volumize(mask_image, F_set)
    p_vol = volumize(mask_image, p_set_dict[0])
    log_p_vol = volumize(mask_image, log_p_set)
    one_p_vol = volumize(mask_image, one_p_set)

    base_dir = Path(f"/mnt/NAS2-2/data/SAD_gangnam_MDMR/{smoothness}mm/{subject_group}/{variable_of_interest}/result/")
    base_dir.mkdir(parents=True, exist_ok=True)
    F_file = base_dir / f"pseudo_F_volume.nii.gz"
    p_file = base_dir / f"p_significance_volume.nii.gz"
    log_p_file = base_dir / f"neglog_p_significance_volume.nii.gz"
    one_p_file = base_dir / f"one_minus_p_values.nii.gz"

    base_dir.mkdir(parents=True, exist_ok=True)

    volume_dir = Path(f"/mnt/NAS2-2/data/SAD_gangnam_MDMR/{smoothness}mm/{subject_group}/{variable_of_interest}/temp/volume")
    volume_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0,permutations):
        temp_p_vol = volumize(mask_image, p_set_dict[i])
        temp_p_file = volume_dir / f"p_significance_volume_{i}.nii.gz"
        temp_p_vol.to_filename(temp_p_file)
    F_vol.to_filename(F_file)
    p_vol.to_filename(p_file)
    log_p_vol.to_filename(log_p_file)
    one_p_vol.to_filename(one_p_file)
    if 1 in z_score:
        zvals = pval_to_zval(p_set, permutations)
        z_file = zstat_image(zvals, mask_file, subject_group, variable_of_interest, smoothness)
    else:
        z_file = None

    return F_file, p_file, log_p_file, one_p_file, z_file

def zstat_image(zvals, mask_file, subject_group, variable_of_interest, smoothness):
    mask_image = nb.load(mask_file)

    z_vol = volumize(mask_image, zvals)

    base_dir = Path(f"/mnt/NAS2-2/data/SAD_gangnam_MDMR/{smoothness}mm/{subject_group}/{variable_of_interest}/result/")
    base_dir.mkdir(parents=True, exist_ok=True)
    z_file = base_dir / "zstat.nii.gz"
 
    z_vol.to_filename(z_file)
    return z_file
