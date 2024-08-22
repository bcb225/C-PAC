#!/bin/bash

# Base directory 설정
base_dir="/mnt/NAS2-2/data/SAD_gangnam_resting_2/fMRIPrep_total"

# 처리하려는 피험자 수를 저장할 변수 초기화
total_subjects=0

# 이미 처리된 피험자 수를 저장할 변수 초기화
already_processed_count=0

# 이번에 새로 처리된 피험자 수를 저장할 변수 초기화
processed_count=0

# 모든 피험자 폴더에 대해 반복
for subject_dir in ${base_dir}/sub-*/; do
    # 피험자 코드 추출
    subject_code=$(basename ${subject_dir})
    
    # 세션 폴더 경로 설정
    func_dir="${subject_dir}ses-01/func/"
    
    # 입력 파일 이름 설정 (smoothing이 적용된 파일)
    input_file="${func_dir}${subject_code}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed6mm_bold.nii.gz"
    
    # 출력 파일 이름 설정 (resampling 적용된 파일)
    output_file="${func_dir}${subject_code}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed6mm_resampled4mm_bold.nii.gz"
    
    # 처리하려는 피험자 수 증가
    total_subjects=$((total_subjects + 1))
    
    # 입력 파일이 없는 경우 스킵
    if [ ! -f "${input_file}" ]; then
        echo "Input file not found for ${subject_code}, skipping..."
        continue
    fi
    
    # 출력 파일이 이미 존재하는 경우 스킵
    if [ -f "${output_file}" ]; then
        echo "Resampling already applied for ${subject_code}, skipping..."
        already_processed_count=$((already_processed_count + 1))
        continue
    fi
    
    # 4mm 등방성 해상도로 resampling 적용
    3dresample -input ${input_file} -dxyz 4 4 4 -rmode Cu -prefix ${output_file}
    
    # 결과 출력
    echo "Resampling applied for ${subject_code}, saved as ${output_file}"
    
    # 이번에 새로 처리된 피험자 수 증가
    processed_count=$((processed_count + 1))
done

# 최종 결과 출력
echo "Total subjects intended for processing: ${total_subjects}"
echo "Subjects already processed (skipped): ${already_processed_count}"
echo "Newly processed subjects in this run: ${processed_count}"
