import numpy as np
from tqdm import tqdm
import nibabel as nb
import subprocess
import pandas as pd
import csv
import os 
from pathlib import Path

from nilearn.reporting import get_clusters_table
from nilearn import image
def check_rank(X):
    k    = X.shape[1]
    rank = np.linalg.matrix_rank(X)
    if rank < k:
        print("Rank Deficient Terminating")
        exit(0)
        #raise Exception("matrix is rank deficient (rank %i vs cols %i)" % (rank, k))

def hat(X):
    Q1, _ = np.linalg.qr(X)
    return Q1.dot(Q1.T)

def gower(D):
    n = D.shape[0]
    A = -0.5 * (D ** 2)
    I = np.eye(n, n)
    uno = np.ones((n, 1))
    C = I - (1.0 / n) * uno.dot(uno.T)
    G = C.dot(A).dot(C)
    return G

def gen_h2(x, cols, indexperm):
    H = gen_h(x, cols, indexperm)
    other_cols = [i for i in range(x.shape[1]) if i not in cols]
    Xj = x[:,other_cols]
    H2 = H - hat(Xj)
    return H2

def permute_design(x, cols, indexperm):
    Xj = x.copy()
    Xj[:, cols] = Xj[indexperm][:, cols]
    return Xj

def gen_h(x, cols, indexperm):
    x = permute_design(x, cols, indexperm)
    H = hat(x)
    return H

def gen_h2_perms(x, cols, perms):
    nperms, nobs = perms.shape
    H2perms = np.zeros((nobs**2, nperms))
    for i in range(nperms):
        H2 = gen_h2(x, cols, perms[i,:])
        H2perms[:,i] = H2.flatten()

    return H2perms

def gen_ih_perms(x, cols, perms):
    nperms, nobs = perms.shape
    I = np.eye(nobs, nobs)

    IHperms = np.zeros((nobs ** 2, nperms))
    for i in range(nperms):
        IH = I - gen_h(x, cols, perms[i, :])
        IHperms[:, i] = IH.flatten()

    return IHperms

def calc_ssq_fast(Hs, Gs, transpose=True):
    if transpose:
        ssq = Hs.T.dot(Gs)
    else:
        ssq = Hs.dot(Gs)
    return ssq

def ftest_fast(Hs, IHs, Gs, df_among, df_resid, **ssq_kwrds):
    SS_among = calc_ssq_fast(Hs, Gs, **ssq_kwrds)
    SS_resid = calc_ssq_fast(IHs, Gs, **ssq_kwrds)
    F = (SS_among / df_among) / (SS_resid / df_resid)
    return F

def mdmr(D, X, columns, permutations, mask_file, base_dir, mode):
    mask_image = nb.load(mask_file)
    p_file = base_dir / f"p_significance_volume.nii.gz"
    # Check if the file exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Base directory {base_dir} created.")
    else:
        print(f"Base directory {base_dir} already exists.")

    check_rank(X)
    
    subjects = X.shape[0]
    if subjects != D.shape[1]:
        raise Exception("# of subjects incompatible between X and D")
    
    voxels = D.shape[0]
    Gs = np.zeros((subjects ** 2, voxels))
    for di in range(voxels):
        Gs[:, di] = gower(D[di]).flatten()
    
    X1 = np.hstack((np.ones((subjects, 1)), X))
    columns = columns.copy() #removed a +1

    regressors = X1.shape[1]

    permutation_indexes = np.zeros((permutations, subjects), dtype=int)
    permutation_indexes[0, :] = range(subjects)
    for i in range(1, permutations):
        permutation_indexes[i,:] = np.random.permutation(subjects)
    
    H2perms = gen_h2_perms(X1, columns, permutation_indexes)
    IHperms = gen_ih_perms(X1, columns, permutation_indexes)

    df_among = len(columns)
    df_resid = subjects - regressors

    F_perms = ftest_fast(H2perms, IHperms, Gs, df_among, df_resid)

    """p_vals = (F_perms[1:, :] >= F_perms[0, :]) \
                .sum(axis=0) \
                .astype('float')
    p_vals /= permutations"""

    # Initialize an array to store p-values for each permutation
    all_p_vals = np.zeros_like(F_perms)

    # Loop through each F_perms[i, :] as the reference, with tqdm for progress display
    for i in tqdm(range(F_perms.shape[0]), desc="Calculating p-values"):
        # Calculate p-values for F_perms[i, :] against all other permutations
        p_vals = (F_perms[np.arange(F_perms.shape[0]) != i, :] >= F_perms[i, :]) \
                    .sum(axis=0) \
                    .astype('float')
        p_vals /= F_perms.shape[0]  # Adjust the division since we are excluding the current permutation

        # Store the computed p-values for this permutation
        all_p_vals[i, :] = p_vals
        if i == 0:
            print("Saving First Result")
            p_vol = volumize(mask_image, p_vals)
            p_vol.to_filename(p_file)
            evaluate_cluster_size(base_dir, mode, mask_image, cut_off = 15)
    return F_perms[0, :], all_p_vals
def evaluate_cluster_size(base_dir, mode, mask_img, cut_off):
    p_file = base_dir / "p_significance_volume.nii.gz"
    input_file= p_file
    p_img = image.load_img(input_file)
    p_data = p_img.get_fdata()
    one_minus_p_data = 1 - p_data
    one_minus_p_img = image.new_img_like(p_img, one_minus_p_data)
    masked_one_minus_p_img = image.math_img("img1 * img2", img1=one_minus_p_img, img2=mask_img)
    table = get_clusters_table(masked_one_minus_p_img, stat_threshold=0.995, cluster_threshold=0)
    # Check if the file exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Base directory {base_dir} created.")
    else:
        print(f"Base directory {base_dir} already exists.")
    
    if not table.empty:
        table['Cluster Size (mm3)'] = pd.to_numeric(table['Cluster Size (mm3)'], errors='coerce')
        max_cluster_size = table["Cluster Size (mm3)"].max()
        # NaN 값이 있는 경우 0으로 처리
        if pd.isna(max_cluster_size):
            max_cluster_size = 0
    else:
        max_cluster_size = 0
    if max_cluster_size < cut_off * 64:
        print("Significant Voxel Not Found... Early Terminating...")
        exit(0)
    else:
        with open('./MDMR_log.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)  # Add quoting to handle special characters
            writer.writerow([str(base_dir)])  # Store the path as a string inside a list
            print("Log written to ./MDMR_log.csv")
        if mode == "scan":
            print("Current mode is scanning mode, detailed analysis requires full mode parameter")
            print("Early stopping for fast scan. Please run full mode in later session.")
            exit(0)
        elif mode == "full":
            print("Significant Voxel Found!!! Keep analyzing...")
        
def volumize(mask_image, data):
    mask_data = mask_image.get_fdata().astype('bool')
    volume = np.zeros_like(mask_data, dtype=data.dtype)
    volume[np.where(mask_data == True)] = data
    return nb.Nifti1Image(
        volume,
        header=mask_image.header,
        affine=mask_image.affine
    )