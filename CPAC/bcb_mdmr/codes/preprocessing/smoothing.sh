#!/bin/bash

# Base directory 설정
base_dir="/mnt/NAS2-2/data/SAD_gangnam_resting_2/fMRIPrep_total"

# 모든 피험자 폴더에 대해 반복
for subject_dir in ${base_dir}/sub-*/; do
    # 피험자 코드 추출
    subject_code=$(basename ${subject_dir})
    
    # 세션 폴더 경로 설정
    func_dir="${subject_dir}ses-01/func/"
    
    # 입력 파일 이름 설정
    input_file="${func_dir}${subject_code}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    
    # 출력 파일 이름 설정 (smoothing 적용된 파일)
    output_file="${func_dir}${subject_code}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed6mm_bold.nii.gz"
    
    # 출력 파일이 이미 존재하는 경우 스킵
    if [ -f "${output_file}" ]; then
        echo "Smoothing already applied for ${subject_code}, skipping..."
        continue
    fi
    
    # 8mm smoothing 적용
    3dBlurToFWHM -FWHM 6 -input ${input_file} -prefix ${output_file}
    
    # 결과 출력
    echo "Smoothing applied for ${subject_code}, saved as ${output_file}"
done
