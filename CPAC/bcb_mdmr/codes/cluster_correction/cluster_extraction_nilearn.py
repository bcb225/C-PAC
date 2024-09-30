import pandas as pd
from nilearn.reporting import get_clusters_table
from nilearn import image
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import interp1d
import argparse
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import sys
sys.path.append('../mdmr/')
from DataHandler import DataHandler  

def process_nii_file(temp_dir, permut_index, mask_img):
    nii_file = temp_dir / "volume" / f"p_significance_volume_{permut_index}.nii.gz"
    cluster_report_file = temp_dir / "cluster_report" / f"cluster_report_{permut_index}.csv"
    try:
        # Create the directory for cluster reports if it doesn't exist
        cluster_report_dir = temp_dir / "cluster_report"
        cluster_report_dir.mkdir(parents=True, exist_ok=True)
        # p-value 이미지 로드
        p_img = image.load_img(nii_file)
        p_data = p_img.get_fdata()

        # 1 - p 계산
        one_minus_p_data = 1 - p_data
        one_minus_p_img = image.new_img_like(p_img, one_minus_p_data)

        # 마스크 적용
        masked_one_minus_p_img = image.math_img("img1 * img2", img1=one_minus_p_img, img2=mask_img)

        # 클러스터 테이블 생성 (클러스터 크기 추출)
        table = get_clusters_table(masked_one_minus_p_img, stat_threshold=0.995, cluster_threshold=0)

        # 클러스터 크기를 숫자로 변환하여 최대 클러스터 크기 추출
        if not table.empty:
            table['Cluster Size (mm3)'] = pd.to_numeric(table['Cluster Size (mm3)'], errors='coerce')
            max_cluster_size = table["Cluster Size (mm3)"].max()
            # NaN 값이 있는 경우 0으로 처리
            table.to_csv(cluster_report_file)
            if pd.isna(max_cluster_size):
                max_cluster_size = 0
        else:
            max_cluster_size = 0

        return max_cluster_size

    except Exception as e:
        print(f"Error processing file {nii_file}: {e}")
        return 0

def max_cluster_distribution(temp_dir, mask_img, alpha_level, permutation):
    max_voxel_list = []

    permutations = range(permutation)

    # 병렬처리
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_nii_file, temp_dir, permut_index, mask_img): permut_index for permut_index in permutations}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing NIfTI files"):
            max_cluster_size = future.result()
            max_voxel_list.append(max_cluster_size)

    # 리스트 정렬
    sorted_voxel_list = np.sort(max_voxel_list)
    #print(sorted_voxel_list)
    # x축은 0과 1 사이의 누적확률, y축은 클러스터 크기
    cumulative_prob = np.linspace(0, 1, len(sorted_voxel_list))

    # scipy의 interp1d 함수를 사용하여 연속적인 값을 추정
    interpolation_function = interp1d(cumulative_prob, sorted_voxel_list, kind='linear', fill_value="extrapolate")

    # alpha 레벨에 해당하는 값 추정 (e.g., 0.05)
    threshold_cluster_size = interpolation_function(1 - alpha_level)

    print(f"Alpha level of {alpha_level*100}% cluster size: {threshold_cluster_size}")
    return threshold_cluster_size

def analyze_clusters(group, variable, smoothness, mask_img, vox_threhold, cluster_threshold_alpha):
    nas_dir = Path("/mnt/NAS2-2/data/")
    MDMR_output_dir = nas_dir / "SAD_gangnam_MDMR"

    mask_img_path = Path("../../template/all_final_group_mask_6mm.nii.gz")
    result_dir = MDMR_output_dir / f"{smoothness}mm" / str(group) / variable / "result"
    temp_dir = MDMR_output_dir / f"{smoothness}mm" / str(group) / variable / "temp"
    p_volume_path = result_dir / "p_significance_volume.nii.gz"

    # 마스크 및 p값 파일 로드
    mask_img = image.load_img(mask_img_path)
    p_img = image.load_img(p_volume_path)
    p_data = p_img.get_fdata()

    # 1 - p 계산
    one_minus_p_data = 1 - p_data
    one_minus_p_img = image.new_img_like(p_img, one_minus_p_data)

    # 마스크 적용 (apply_mask 대신 직접 마스크 이미지로 마스킹)
    masked_one_minus_p_img = image.math_img("img1 * img2", img1=one_minus_p_img, img2=mask_img)

    # 나중에 permuatation 15000으로 수정해야함.
    cluster_threshold = max_cluster_distribution(
        temp_dir=temp_dir,mask_img=mask_img, alpha_level=cluster_threshold_alpha, permutation=150
        )
    
    #cluster_threshold = 64 * 16
    cluster_threshold_vox = cluster_threshold / 64
    
    print(cluster_threshold_vox)
    one_minus_p_threshold = 1-vox_threhold
    # 클러스터 테이블 생성
    table, label_maps = get_clusters_table(masked_one_minus_p_img, stat_threshold=one_minus_p_threshold, cluster_threshold=cluster_threshold_vox, return_label_maps=True)
    table.set_index("Cluster ID", drop=True)
    table["cluster_threshold"] = cluster_threshold

    cluster_report_filename = result_dir / "significant_cluster_report.csv"

    # 각 클러스터의 x, y, z 좌표로 R에서 mni_to_region_name 호출
    ro.r('library(label4MRI)')
    
    aal_labels = []
    aal_distances = []
    ba_labels = []
    ba_distances = []

    for _, row in table.iterrows():
        x, y, z = row['X'], row['Y'], row['Z']
        result = ro.r(f'mni_to_region_name(x = {x}, y = {y}, z = {z})')
        parsed = parse_mni_to_region_name(result)

        # AAL 정보 저장
        aal_labels.append(parsed['aal']['label'])
        aal_distances.append(parsed['aal']['distance'])

        # BA 정보 저장
        ba_labels.append(parsed['ba']['label'])
        ba_distances.append(parsed['ba']['distance'])

    # 결과를 테이블에 추가
    table["AAL Label"] = aal_labels
    table["AAL Distance"] = aal_distances
    table["BA Label"] = ba_labels
    table["BA Distance"] = ba_distances

    # 최종 테이블을 CSV로 저장
    table.to_csv(cluster_report_filename)
    
    #print(table)
    return table, label_maps

def parse_mni_to_region_name(result):
    parsed_result = {}
    
    # Parse AAL information
    aal_distance = result.rx2('aal.distance')[0]
    aal_label = result.rx2('aal.label')[0]
    
    parsed_result['aal'] = {
        'distance': aal_distance,
        'label': aal_label
    }
    
    # Parse BA information
    ba_distance = result.rx2('ba.distance')[0]
    ba_label = result.rx2('ba.label')[0]
    
    parsed_result['ba'] = {
        'distance': ba_distance,
        'label': ba_label
    }
    
    return parsed_result
def extract_cluster_mask(group, variable, smoothness, table, label_maps, mask_img):
    from nilearn import image
    import numpy as np

    # Define directories
    nas_dir = Path("/mnt/NAS2-2/data/")
    MDMR_output_dir = nas_dir / "SAD_gangnam_MDMR"
    result_dir = MDMR_output_dir / f"{smoothness}mm" / str(group) / variable / "result"
    
    # Load the label image (assume label_maps is a single NIfTI image)
    label_img = label_maps[0]  # Since label_maps is a list with a single image
    label_data = label_img.get_fdata()

    # Apply the mask to the label data
    mask_data = mask_img.get_fdata()
    masked_label_data = label_data * mask_data

    # Iterate over unique labels (excluding background)
    unique_labels = np.unique(masked_label_data)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    for label in unique_labels:
        # Extract the region corresponding to the current label
        region_data = (masked_label_data == label).astype(np.float32)

        # Create a new image using nilearn's new_img_like
        region_img = image.new_img_like(label_img, region_data)

        # Query the table for the corresponding label's anatomical name
        region_info = table[table['Cluster ID'] == label]
        if not region_info.empty:
            aal_label = region_info.iloc[0]["AAL Label"]  # Extract AAL label name

            # Save the region with the anatomical name as the file name
            output_filename = result_dir / f"MDMR_significant_aal({aal_label.replace(' ', '_')})_label({int(label)}).nii.gz"
            region_img.to_filename(str(output_filename))
            print(f"Saved region {aal_label} (label {label}) to {output_filename}")
def main():
    parser = argparse.ArgumentParser(description="Analyze clusters with specified parameters.")
    
    # Add arguments for group, variable, smoothness, vox_threhold, and cluster_threshold_alpha
    parser.add_argument('--regressor_file', type=str, required=True, help="Group name (e.g., 'gangnam_total').")
    parser.add_argument('--smoothness', type=int, default=6, help="Smoothness value (default: 6).")
    parser.add_argument('--vox_threhold', type=float, default=0.005, help="Voxel threshold (default: 0.005).")
    parser.add_argument('--cluster_threshold_alpha', type=float, default=0.05, help="Cluster threshold alpha level (default: 0.05).")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize DataHandler to retrieve group number
    data_handler = DataHandler()

    # Retrieve the group number for the provided regressor file
    group_num = data_handler.get_subject_group(args.regressor_file)
    variable_name = data_handler.get_variable(args.regressor_file)
    
    mask_img_path = Path("../../template/all_final_group_mask_6mm.nii.gz")
    mask_img = image.load_img(mask_img_path)
    
    # Call analyze_clusters with the parsed arguments
    table, label_maps = analyze_clusters(
        group=group_num, 
        variable=variable_name, 
        smoothness=args.smoothness, 
        mask_img = mask_img,
        vox_threhold=args.vox_threhold, 
        cluster_threshold_alpha=args.cluster_threshold_alpha
    )
    #label_maps[0].to_filename("label_maps.nii.gz")
    extract_cluster_mask(
        group=group_num, 
        variable=variable_name, 
        smoothness=args.smoothness, 
        table = table,
        label_maps = label_maps,
        mask_img = mask_img
    )
    print(table)
    #print(len(label_maps))

if __name__ == "__main__":
    main()