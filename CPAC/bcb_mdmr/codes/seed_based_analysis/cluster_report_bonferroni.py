import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
from multiprocessing import Pool, cpu_count
import subprocess

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
    
    input_file= f"{result_dir}/bonferroni_correted_z_map.nii.gz"
    output_file=f"{result_dir}/bonferroni_cluster_report.tsv"
    
    
    subprocess.run(['fsl-cluster', '-i', input_file, '-t', '0.5'], stdout=open(output_file, 'w'), check=True)