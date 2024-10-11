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
import re
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

def analyze_clusters(group, variable, smoothness, mask_img, vox_threhold, cluster_threshold_alpha):
    nas_dir = Path("/mnt/NAS2-2/data/")
    MDMR_output_dir = nas_dir / "SAD_gangnam_MDMR"

    mask_img_path = Path("../../template/all_final_group_mask_6mm.nii.gz")
    result_dir = MDMR_output_dir / f"{smoothness}mm" / str(group) / variable / "result"
    temp_dir = MDMR_output_dir / f"{smoothness}mm" / str(group) / variable / "temp"
    p_volume_path = result_dir / "p_significance_volume.nii.gz"

    # Load mask and p-value images
    mask_img = image.load_img(mask_img_path)
    p_img = image.load_img(p_volume_path)
    p_data = p_img.get_fdata()

    # Compute 1 - p
    one_minus_p_data = 1 - p_data
    one_minus_p_img = image.new_img_like(p_img, one_minus_p_data)

    # Apply mask
    masked_one_minus_p_img = image.math_img("img1 * img2", img1=one_minus_p_img, img2=mask_img)

    # Compute cluster threshold
    cluster_threshold = max_cluster_distribution(
        temp_dir=temp_dir, mask_img=mask_img, alpha_level=cluster_threshold_alpha, permutation=15000
    )

    cluster_threshold_vox = cluster_threshold / 64

    one_minus_p_threshold = 1 - vox_threhold

    # Generate cluster table and label maps
    table, label_maps = get_clusters_table(
        masked_one_minus_p_img,
        stat_threshold=one_minus_p_threshold,
        cluster_threshold=cluster_threshold_vox,
        return_label_maps=True
    )

    # Initialize R and load the required library
    ro.r('library(label4MRI)')

    # Parse AAL and BA labels for each cluster peak
    aal_labels = []
    aal_distances = []
    ba_labels = []
    ba_distances = []

    for _, row in table.iterrows():
        x, y, z = row['X'], row['Y'], row['Z']
        result = ro.r(f'mni_to_region_name(x = {x}, y = {y}, z = {z})')
        parsed = parse_mni_to_region_name(result)

        # AAL information
        aal_labels.append(parsed['aal']['label'])
        aal_distances.append(parsed['aal']['distance'])

        # BA information
        ba_labels.append(parsed['ba']['label'])
        ba_distances.append(parsed['ba']['distance'])

    # Add the labels to the table
    table["AAL Label"] = aal_labels
    table["AAL Distance"] = aal_distances
    table["BA Label"] = ba_labels
    table["BA Distance"] = ba_distances

    # Save the table
    cluster_report_filename = result_dir / "significant_cluster_report.csv"
    table.to_csv(cluster_report_filename, index=False)

    return table, label_maps

# 추가된 함수: 구형 마스크 생성
def create_sphere_mask(center_coords, radius, shape, affine):
    import numpy as np
    import nibabel

    # 모든 복셀 인덱스 생성
    xx, yy, zz = np.meshgrid(np.arange(shape[0]),
                             np.arange(shape[1]),
                             np.arange(shape[2]),
                             indexing='ij')

    # 배열을 평탄화
    voxel_coords = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # 복셀 좌표를 MNI 좌표로 변환
    mni_coords = nibabel.affines.apply_affine(affine, voxel_coords)

    # 중심으로부터의 거리 계산
    distances = np.sqrt(np.sum((mni_coords - center_coords) ** 2, axis=1))

    # 마스크 생성
    mask_data = (distances <= radius).astype(np.int8)
    mask_data = mask_data.reshape(shape)

    return mask_data

def extract_cluster_mask(group, variable, smoothness, table, label_maps, mask_img):
    from nilearn import image
    import numpy as np
    import pandas as pd
    import nibabel
    import re

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

    # Ensure 'ClusterID_numeric' column exists in the table
    if 'ClusterID_numeric' not in table.columns:
        table['ClusterID_numeric'] = table['Cluster ID'].apply(
            lambda x: int(re.match(r'\d+', str(x)).group())
        )

    # Iterate over unique labels (excluding background)
    unique_labels = np.unique(masked_label_data)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    for label in unique_labels:
        # Extract the region corresponding to the current label
        region_data = (masked_label_data == label).astype(np.float32)

        # Create a new image using nilearn's new_img_like
        region_img = image.new_img_like(label_img, region_data)

        # Run get_clusters_table on region_img to get center of mass
        cluster_table = get_clusters_table(
            region_img, stat_threshold=0.5, cluster_threshold=0
        )

        if not cluster_table.empty:
            # 피크 통계 좌표
            peak_x = cluster_table.iloc[0]['X']
            peak_y = cluster_table.iloc[0]['Y']
            peak_z = cluster_table.iloc[0]['Z']

            # Use R functions to get labels for these coordinates
            result = ro.r(f'mni_to_region_name(x = {peak_x}, y = {peak_y}, z = {peak_z})')
            parsed = parse_mni_to_region_name(result)

            # AAL information
            com_aal_label = parsed['aal']['label']
            com_aal_distance = parsed['aal']['distance']

            # BA information
            com_ba_label = parsed['ba']['label']
            com_ba_distance = parsed['ba']['distance']

            # Update the main table
            cluster_id_numeric = int(label)
            rows_to_update = table['ClusterID_numeric'] == cluster_id_numeric

            table.loc[rows_to_update, 'Center of Mass X'] = peak_x
            table.loc[rows_to_update, 'Center of Mass Y'] = peak_y
            table.loc[rows_to_update, 'Center of Mass Z'] = peak_z
            table.loc[rows_to_update, 'Center of Mass AAL Label'] = com_aal_label
            table.loc[rows_to_update, 'Center of Mass AAL Distance'] = com_aal_distance
            table.loc[rows_to_update, 'Center of Mass BA Label'] = com_ba_label
            table.loc[rows_to_update, 'Center of Mass BA Distance'] = com_ba_distance
        else:
            print(f"No clusters found in region for label {label}")
            continue  # Skip to the next label if no clusters found

        # Query the table for the corresponding label's center of mass AAL label
        region_info = table[table['ClusterID_numeric'] == int(label)]
        if not region_info.empty:
            com_aal_label = region_info.iloc[0]["Center of Mass AAL Label"]  # Use center of mass AAL label

            # Save the region with the center of mass AAL label as the file name
            # Ensure the output directory exists
            output_dir = result_dir / "cluster_masks"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Clean the AAL label to create a valid filename
            aal_label_clean = re.sub(r'[^a-zA-Z0-9_]', '_', com_aal_label)

            output_filename = output_dir / f"MDMR_significant_aal({aal_label_clean})_label({int(label)}).nii.gz"
            region_img.to_filename(str(output_filename))
            print(f"Saved region {com_aal_label} (label {label}) to {output_filename}")

            # 피크 통계 좌표 가져오기
            peak_coords = (region_info.iloc[0]['X'], region_info.iloc[0]['Y'], region_info.iloc[0]['Z'])

            # 중심 좌표 가져오기 (만약 중심 좌표를 별도로 계산했다면 사용)
            com_coords = (region_info.iloc[0]['Center of Mass X'], region_info.iloc[0]['Center of Mass Y'], region_info.iloc[0]['Center of Mass Z'])

            # 구형 마스크 생성
            radius = 6  # 반경 6mm
            shape = label_img.shape[:3]
            affine = label_img.affine

            # 피크 통계 좌표에 대한 구형 마스크
            peak_sphere_mask_data = create_sphere_mask(peak_coords, radius, shape, affine)
            peak_sphere_mask_img = image.new_img_like(label_img, peak_sphere_mask_data)

            # 중심 좌표에 대한 구형 마스크
            com_sphere_mask_data = create_sphere_mask(com_coords, radius, shape, affine)
            com_sphere_mask_img = image.new_img_like(label_img, com_sphere_mask_data)

            # 마스크 저장
            # 피크 통계 구형 마스크
            peak_output_filename = output_dir / f"peak_stat_sphere_aal_{aal_label_clean}_label_{int(label)}.nii.gz"
            peak_sphere_mask_img.to_filename(str(peak_output_filename))
            print(f"Saved peak stat sphere mask for region {com_aal_label} (label {label}) to {peak_output_filename}")

            # 중심 좌표 구형 마스크
            com_output_filename = output_dir / f"center_of_mass_sphere_aal_{aal_label_clean}_label_{int(label)}.nii.gz"
            com_sphere_mask_img.to_filename(str(com_output_filename))
            print(f"Saved center of mass sphere mask for region {com_aal_label} (label {label}) to {com_output_filename}")

        else:
            print(f"No region info found for label {label}")

    # After updating the table, save it to CSV
    cluster_report_filename = result_dir / "significant_cluster_report.csv"
    table.to_csv(cluster_report_filename, index=False)
    print(f"Updated cluster report saved to {cluster_report_filename}")

def main():
    parser = argparse.ArgumentParser(description="Analyze clusters with specified parameters.")
    
    # Add arguments
    parser.add_argument('--regressor_file', type=str, required=True, help="Regressor file path.")
    parser.add_argument('--smoothness', type=int, default=6, help="Smoothness value (default: 6).")
    parser.add_argument('--vox_threhold', type=float, default=0.005, help="Voxel threshold (default: 0.005).")
    parser.add_argument('--cluster_threshold_alpha', type=float, default=0.05, help="Cluster threshold alpha level (default: 0.05).")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize DataHandler to retrieve group number
    data_handler = DataHandler()

    # Retrieve the group number and variable name
    group_num = data_handler.get_subject_group(args.regressor_file)
    variable_name = data_handler.get_variable(args.regressor_file)
    
    mask_img_path = Path("../../template/all_final_group_mask_6mm.nii.gz")
    mask_img = image.load_img(mask_img_path)
    
    # Call analyze_clusters with the parsed arguments
    table, label_maps = analyze_clusters(
        group=group_num, 
        variable=variable_name, 
        smoothness=args.smoothness, 
        mask_img=mask_img,
        vox_threhold=args.vox_threhold, 
        cluster_threshold_alpha=args.cluster_threshold_alpha
    )

    # Extract cluster masks and update the table
    extract_cluster_mask(
        group=group_num, 
        variable=variable_name, 
        smoothness=args.smoothness, 
        table=table,
        label_maps=label_maps,
        mask_img=mask_img
    )

    # Print the updated table
    print(table)

if __name__ == "__main__":
    main()
