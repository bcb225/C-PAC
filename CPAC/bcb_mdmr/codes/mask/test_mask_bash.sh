#!/bin/bash

# 파라미터 확인
if [ "$#" -ne 1 ]; then
    echo "사용법: $0 <subject_group>"
    exit 1
fi

# 파라미터 할당
subject_group=$1
input_file="../../input/${subject_group}_code_list.csv"  # CSV 파일 경로
maskdir="../../template/"  # 현재 작업 디렉토리
maskfile="../../template/${subject_group}_group_mask.nii.gz"  # 최종 그룹 마스크 파일명
grey_matter="../../template/tpl-MNI152NLin2009cAsym_space-MNI_res-01_class-GM_probtissue.nii.gz"
resampled_gm="../../template/${subject_group}_resampled_gm.nii.gz"  # 재샘플링된 GM 파일명
threshold_gm="../../template/${subject_group}_threshold_gm.nii.gz"  # 임계값이 적용된 GM 파일명
final_mask="../../template/${subject_group}_final_group_mask.nii.gz"
group_prop_mask="../../template/${subject_group}_group_prop_subjs.nii.gz"
small_final_mask="../../template/${subject_group}_small_final_group_mask.nii.gz"

# 기본 fMRI 파일 경로 패턴
base_path="/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep"
mask_suffix="ses-01/func"

# Create individual subject brain masks
funcpaths=()
echo "CSV 파일 내용 확인:"
while IFS=, read -r subject_code; do
    echo "읽은 subject code: $subject_code"
    func_path="${base_path}/sub-${subject_code}/${mask_suffix}/sub-${subject_code}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    echo "확인 중: ${func_path}"
    if [ -f "${func_path}" ]; then
        funcpaths+=("${func_path}")
    else
        echo "파일을 찾을 수 없습니다: ${func_path}"
    fi
done < "$input_file"

n=${#funcpaths[@]}
if [ $n -eq 0 ]; then
    echo "유효한 fMRI 파일이 없습니다. 종료합니다."
    exit 1
fi
echo "처리할 피험자 수: ${n}"

# 각 피험자의 마스크 생성
for (( i = 0; i < $n; i++ )); do
    func=${funcpaths[$i]}
    mask="${maskdir}/mask${i}.nii.gz"
    echo "생성 중: ${mask} (from ${func})"
    fslmaths ${func} -Tstd -bin ${mask}
done

# Take the mean of the masks
# (i.e., proportion of subjects with value in each voxel)
echo "마스크 평균 계산 중..."
3dMean -prefix ${group_prop_mask} ${maskdir}/mask*.nii.gz

# Optional: concatenate all the subject masks together to view them
echo "마스크 병합 중..."

# Get voxels with all subjects having a value
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

# 특정 위치 범위 내의 복셀만 선택하여 작은 마스크 생성
# 여기서 x, y, z 범위를 지정합니다. 예: x: 20~50, y: 30~60, z: 10~40
echo "작은 그룹 마스크 생성 중..."
3dcalc -a ${final_mask} -expr 'step(a)*step(50-x)*step(x-20)*step(60-y)*step(y-30)*step(40-z)*step(z-10)' -prefix ${small_final_mask}

echo "최종 작은 그룹 마스크가 ${small_final_mask}로 저장되었습니다."

# final_mask의 1인 복셀 개수 출력
echo "final_mask의 1인 복셀 개수:"
3dBrickStat -count -non-zero ${final_mask}

# small_final_mask의 1인 복셀 개수 출력
echo "small_final_mask의 1인 복셀 개수:"
3dBrickStat -count -non-zero ${small_final_mask}

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
