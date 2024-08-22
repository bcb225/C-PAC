import nibabel as nib
import numpy as np
from scipy.ndimage import label
import glob
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 마스크 파일 로드
mask_file = "../../template/gangnam_total_final_group_mask.nii.gz"
mask_img = nib.load(mask_file)
mask_data = mask_img.get_fdata()

# 폴더 내의 모든 p_significance_volume_{i}.nii.gz 파일을 순회
folder_path = "../../output/gangnam_total/LSAS/temp/"  # NIfTI 파일이 있는 폴더 경로
file_pattern = os.path.join(folder_path, "p_significance_volume_*.nii.gz")
file_list = glob.glob(file_pattern)

max_cluster_sizes = []

def process_file(file_path):
    # p-value 데이터 로드
    img = nib.load(file_path)
    data = img.get_fdata()

    # 마스크된 영역에서 p-value threshold 적용 (p < 0.05)
    thresholded_data = (data < 0.05) & (mask_data > 0)

    # binary화된 데이터에 대해 클러스터를 식별
    labeled_array, num_features = label(thresholded_data)

    # 각 클러스터의 크기를 계산하고, 최대 클러스터 크기 반환
    cluster_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
    
    if cluster_sizes:
        max_cluster_size = max(cluster_sizes)
        return max_cluster_size
    else:
        return None

# 20개씩 batch로 묶어 병렬 처리
batch_size = 20
num_batches = len(file_list) // batch_size + int(len(file_list) % batch_size != 0)

with ThreadPoolExecutor(max_workers=batch_size) as executor:
    for i in tqdm(range(num_batches)):
        batch_files = file_list[i*batch_size:(i+1)*batch_size]
        results = list(executor.map(process_file, batch_files))
        # 결과에서 None을 제외하고 max_cluster_sizes에 추가
        max_cluster_sizes.extend([r for r in results if r is not None])

# 최대 클러스터 크기의 분포 구하기
if max_cluster_sizes:
    max_cluster_sizes = np.array(max_cluster_sizes)
    print("Max Cluster Sizes Distribution:", max_cluster_sizes)

    # 5% 알파 레벨에 해당하는 클러스터 크기 계산
    alpha_level = 0.05
    threshold_cluster_size = np.percentile(max_cluster_sizes, 100 * (1 - alpha_level))
    print(f"Alpha level of 5% cluster size: {threshold_cluster_size}")
else:
    print("No clusters found in any of the files.")
