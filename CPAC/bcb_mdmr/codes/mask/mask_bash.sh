#!/bin/bash

# 파라미터 확인
if [ "$#" -ne 2 ]; then
    echo "사용법: $0 <subject_group> <smoothnes>"
    exit 1
fi

# 파라미터 할당
subject_group=$1
smoothness=$2
input_file="../../regressor/${subject_group}_code_list.csv"  # CSV 파일 경로
maskdir="../../template/"  # 현재 작업 디렉토리
maskfile="../../template/${subject_group}_group_mask_${smoothness}mm.nii.gz"  # 최종 그룹 마스크 파일명
grey_matter="../../template/tpl-MNI152NLin2009cAsym_space-MNI_res-01_class-GM_probtissue.nii.gz"
resampled_gm="../../template/${subject_group}_resampled_gm_${smoothness}mm.nii.gz"  # 재샘플링된 GM 파일명
threshold_gm="../../template/${subject_group}_threshold_gm_${smoothness}mm.nii.gz"  # 임계값이 적용된 GM 파일명
final_mask="../../template/${subject_group}_final_group_mask_${smoothness}mm.nii.gz"
group_prop_mask="../../template/${subject_group}_group_prop_subjs_${smoothness}mm.nii.gz"

# 기본 fMRI 파일 경로 패턴
base_path="/mnt/NAS2-2/data/SAD_gangnam_resting_2/fMRIPrep_total"
mask_suffix="ses-01/func"

# Create individual subject brain masks
funcpaths=()
echo "CSV 파일 내용 확인:"
while IFS=, read -r subject_code; do
    func_path="${base_path}/sub-${subject_code}/${mask_suffix}/sub-${subject_code}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed${smoothness}mm_resampled4mm_scrbold.nii.gz"
    if [ -f "${func_path}" ];then
        funcpaths+=("${func_path}")
    else
        echo "파일을 찾을 수 없습니다: ${func_path}"
    fi
done < "$input_file"

n=${#funcpaths[@]}
if [ $n -eq 0 ];then
    echo "유효한 fMRI 파일이 없습니다. 종료합니다."
    exit 1
fi
echo "처리할 피험자 수: ${n}"

# 각 피험자의 마스크 생성
for (( i = 0; i < $n; i++ )); do
    func=${funcpaths[$i]}
    mask="${maskdir}/mask${i}.nii.gz"
    fslmaths ${func} -Tstd -bin ${mask}
    
    # 피험자별 활성화된 복셀 수 계산 및 출력
    voxel_count=$(fslstats ${mask} -V | awk '{print $1}')
    
    if [ "$voxel_count" -le 70000 ]; then
        echo "피험자 ${i+1} (${func}): 활성화된 복셀 수 = ${voxel_count}"
    fi
done

# Take the mean of the masks
echo "마스크 평균 계산 중..."
3dMean -prefix ${group_prop_mask} ${maskdir}/mask*.nii.gz

echo "모든 피험자가 값을 가진 복셀 선택 중..."
3dcalc -a ${group_prop_mask} -expr 'equals(a,1)' -prefix ${maskfile}

# GM 확률 조직 파일을 fMRI 해상도에 맞게 재샘플링
echo "GM 확률 조직 파일을 fMRI 해상도에 맞게 재샘플링 중..."
3dresample -master ${maskfile} -inset ${grey_matter} -prefix ${resampled_gm}

# 재샘플링된 GM 파일에 25% 임계값 적용
echo "재샘플링된 GM 파일에 25% 임계값 적용 중..."
3dcalc -a ${resampled_gm} -expr 'step(a-0.25)' -prefix ${threshold_gm}

# 그룹 마스크와 임계값이 적용된 GM 마스크 결합
echo "그룹 마스크와 임계값이 적용된 GM 마스크 결합 중..."
3dcalc -a ${maskfile} -b ${threshold_gm} -expr 'a*b' -prefix ${final_mask}

echo "최종 그룹 마스크가 ${final_mask}로 저장되었습니다."

# 최종 마스크 파일의 활성화된 복셀 수 계산 및 출력
final_voxel_count=$(fslstats ${final_mask} -V | awk '{print $1}')
echo "최종 그룹 마스크의 활성화된 복셀 수: ${final_voxel_count}"

# 임시 마스크 파일 삭제
for (( i = 0; i < $n; i++ )); do
    mask="${maskdir}/mask${i}.nii.gz"
    if [ -f "$mask" ]; then
        echo "삭제 중: ${mask}"
        rm "$mask"
    else
        echo "파일을 찾을 수 없습니다: ${mask}"
    fi
done
