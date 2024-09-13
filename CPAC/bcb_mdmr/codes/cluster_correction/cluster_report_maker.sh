#!/bin/bash

# Base directory is defined here
base_dir="/mnt/NAS2-2/data/SAD_gangnam_MDMR/"

# Default values for parameters (optional)
group=""
variable=""
permutations=0
smoothness=-1

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --group) group="$2"; shift ;;
        --variable) variable="$2"; shift ;;
        --permutations) permutations="$2"; shift ;;
        --smoothness) smoothness="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if all required parameters are set
if [[ -z "$group" || -z "$variable" || -z "$permutations" || -z "$smoothness" ]]; then
    echo "Error: All parameters --group, --variable, --permutations. --smoothness must be provided."
    exit 1
fi

# Function to process a single permutation
process_permutation() {
    local i=$1
    local group=$2
    local variable=$3
    local base_dir=$4
    local smoothness=$5
    input_file="${base_dir}/${smoothness}mm/${group}/${variable}/temp/volume/p_significance_volume_${i}.nii.gz"
    mask_file="${base_dir}/${smoothness}mm/${group}/${variable}/temp/mask/mask_${i}.nii.gz"
    output_file="${base_dir}/${smoothness}mm/${group}/${variable}/temp/cluster_report/cluster_report_${i}.txt"

    mkdir -p "${base_dir}/${smoothness}mm/${group}/${variable}/temp/mask"
    mkdir -p "${base_dir}/${smoothness}mm/${group}/${variable}/temp/cluster_report"

    if [[ ! -f $input_file ]]; then
        echo "Error: Input file $input_file does not exist. Skipping this permutation."
        return 1
    fi

    fslmaths $input_file -uthr 0.005 -bin $mask_file
    if [[ $? -ne 0 ]]; then
        echo "Error: fslmaths failed for $input_file. Skipping this permutation."
        return 1
    fi

    fsl-cluster -i $mask_file -t 0.5 > $output_file
    if [[ $? -ne 0 ]]; then
        echo "Error: fsl-cluster failed for $mask_file. Skipping this permutation."
        return 1
    fi

    return 0
}

export -f process_permutation

# Run the permutations in parallel with progress bar
seq 0 $((permutations - 1)) | parallel -j 20 --bar process_permutation {} "$group" "$variable" "$base_dir" "$smoothness"

echo "모든 permutation이 그룹 $group과 변수 $variable에 대해 처리되었습니다."
