import numpy as np
from tqdm import tqdm
import nibabel as nb
import subprocess
import pandas as pd
import csv

def check_rank(X):
    k    = X.shape[1]
    rank = np.linalg.matrix_rank(X)
    if rank < k:
        raise Exception("matrix is rank deficient (rank %i vs cols %i)" % (rank, k))

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

def mdmr(D, X, columns, permutations, mask_file, base_dir):
    mask_image = nb.load(mask_file)
    p_file = base_dir / f"p_significance_volume.nii.gz"
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
            evaluate_cluster_size(base_dir, cut_off = 15)
    return F_perms[0, :], all_p_vals
def evaluate_cluster_size(base_dir, cut_off):
    p_file = base_dir / "p_significance_volume.nii.gz"
    input_file= p_file
    
    mask_file= base_dir/"mask.nii.gz"
    output_file=base_dir/"cluster_report.tsv"
    subprocess.run(['fslmaths', input_file, '-uthr', '0.005', '-bin', mask_file], check=True)
    subprocess.run(['fsl-cluster', '-i', mask_file, '-t', '0.5'], stdout=open(output_file, 'w'), check=True)
    
    try:
        report = pd.read_csv(output_file, delimiter='\t')
        if 'Voxels' in report.columns and len(report) > 0:
            max_voxel_count = report['Voxels'].values[0]
        else:
            print(f"Warning: File {output_file} is empty or does not contain 'Voxels' column.")
            max_voxel_count = None
    except Exception as e:
        print(f"Error reading {output_file}: {e}")
        max_voxel_count = None
    print(f"Max Voxel Count: {max_voxel_count}")
    if max_voxel_count < cut_off:
        print("Significant Voxel Not Found... Early Terminating...")
        exit(0)
    else:
        print("Significant Voxel Found!!! Keep analyzing...")
        with open('./MDMR_log.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)  # Add quoting to handle special characters
            writer.writerow([str(base_dir)])  # Store the path as a string inside a list
            print("Log written to ./MDMR_log.csv")
        
def volumize(mask_image, data):
    mask_data = mask_image.get_fdata().astype('bool')
    volume = np.zeros_like(mask_data, dtype=data.dtype)
    volume[np.where(mask_data == True)] = data
    return nb.Nifti1Image(
        volume,
        header=mask_image.header,
        affine=mask_image.affine
    )