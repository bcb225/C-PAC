import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
from multiprocessing import Pool, cpu_count
import subprocess


def run_permutation_test(args):
    variable, permutation_num, smoothness, group = args
    temp_dir = f"/mnt/NAS2-2/data/SAD_gangnam_seed_based_analysis/{smoothness}mm/p_map/{group}/{variable}/temp/"
    input_file= f"{temp_dir}/volume/{permutation_num}_pvalue_map.nii.gz"
    mask_dir= f"{temp_dir}/mask"
    output_dir=f"{temp_dir}/cluster_report/"
    
    mask_file= f"{mask_dir}/mask_{permutation_num}.nii.gz"
    output_file=f"{output_dir}/cluster_report_{permutation_num}.tsv"
    
    subprocess.run(['fslmaths', input_file, '-uthr', '0.005', '-bin', mask_file], check=True)
    subprocess.run(['fsl-cluster', '-i', mask_file, '-t', '0.5'], stdout=open(output_file, 'w'), check=True)
def run_batch_cluster_correction(variable, start_idx, end_idx, smoothness, group):
    args_list = [(variable, i, smoothness, group) for i in range(start_idx, end_idx + 1)]
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
    result_dir = f"{seed_anal_dir}/{args.smoothness}mm/p_map/{args.group}/{args.variable}/result/"
    temp_dir = f"{seed_anal_dir}/{args.smoothness}mm/p_map/{args.group}/{args.variable}/temp/"
    
    input_file= f"{result_dir}/pvalue_map.nii.gz"
    mask_dir= f"{temp_dir}/mask"
    output_dir=f"{temp_dir}/cluster_report/"
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    mask_file= f"{mask_dir}/mask_0.nii.gz"
    output_file=f"{output_dir}/cluster_report_0.tsv"
    
    subprocess.run(['fslmaths', input_file, '-uthr', '0.0005', '-bin', mask_file], check=True)
    subprocess.run(['fsl-cluster', '-i', mask_file, '-t', '0.5'], stdout=open(output_file, 'w'), check=True)

    total_permutations = 15000
    num_batches = (total_permutations + args.batch_size - 1) // args.batch_size

    for batch in range(num_batches):
        start_idx = batch * args.batch_size + 1
        end_idx = min((batch + 1) * args.batch_size, total_permutations)
        print(f"Running batch {batch + 1}/{num_batches} (permutations {start_idx}-{end_idx})")
        run_batch_cluster_correction(args.variable, start_idx, end_idx, args.smoothness, args.group)

    print("All permutations completed.")