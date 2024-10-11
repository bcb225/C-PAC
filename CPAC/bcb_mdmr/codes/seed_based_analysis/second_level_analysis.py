import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
from nilearn.reporting import get_clusters_table
from nilearn.maskers import NiftiMasker
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn import image
import rpy2.robjects as ro
import re
import sys
import warnings

# nilearn의 특정 워닝을 무시하는 설정
warnings.filterwarnings("ignore", message=".*At least one of the \(sub\)peaks falls outside of the cluster body.*")
warnings.filterwarnings("ignore", message=".*Setting an item of incompatible dtype is deprecated.*")
warnings.filterwarnings("ignore", message=".*Design matrix is singular.*")
warnings.filterwarnings("ignore", message="Attention: No clusters with stat higher than")

sys.path.append('../mdmr/')
from DataHandler import DataHandler

# 구형 마스크 생성 함수
def create_sphere_mask(center_coords, radius, shape, affine):
    import numpy as np
    import nibabel

    xx, yy, zz = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    voxel_coords = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    mni_coords = nibabel.affines.apply_affine(affine, voxel_coords)
    distances = np.sqrt(np.sum((mni_coords - center_coords) ** 2, axis=1))
    mask_data = (distances <= radius).astype(np.int8)
    mask_data = mask_data.reshape(shape)

    return mask_data

# R에서 MNI 좌표로부터 지역 이름을 추출하는 함수를 호출하고 결과를 파싱하는 함수
def parse_mni_to_region_name(result):
    parsed_result = {}
    aal_distance = result.rx2('aal.distance')[0]
    aal_label = result.rx2('aal.label')[0]
    parsed_result['aal'] = {'distance': aal_distance, 'label': aal_label}
    ba_distance = result.rx2('ba.distance')[0]
    ba_label = result.rx2('ba.label')[0]
    parsed_result['ba'] = {'distance': ba_distance, 'label': ba_label}
    
    return parsed_result

# ROI에서 클러스터 마스크를 추출하는 함수
def extract_cluster_mask(group, variable, smoothness, roi, table, label_maps, mask_img, regressor_name, other_regressor_file_path, other_variable_name):
    from nilearn import image
    import numpy as np
    import pandas as pd
    import re
    from pathlib import Path
    import nibabel

    # 디렉토리 경로 설정
    nas_dir = Path("/mnt/NAS2-2/data/")
    seed_anal_dir = nas_dir / "SAD_gangnam_seed_based_analysis"
    
    # 결과를 저장할 디렉토리를 수정된 구조로 생성
    result_dir = seed_anal_dir / f"{smoothness}mm" / "second_level_results" / str(group) / variable / roi / other_regressor_file_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)

    label_img = label_maps[0]
    label_data = label_img.get_fdata()
    mask_data = mask_img.get_fdata()
    masked_label_data = label_data * mask_data

    if 'ClusterID_numeric' not in table.columns:
        table['ClusterID_numeric'] = table['Cluster ID'].apply(lambda x: int(re.match(r'\d+', str(x)).group()))

    unique_labels = np.unique(masked_label_data)
    unique_labels = unique_labels[unique_labels != 0]

    for label in unique_labels:
        region_data = (masked_label_data == label).astype(np.float32)
        region_img = image.new_img_like(label_img, region_data)
        cluster_table = get_clusters_table(region_img, stat_threshold=0.5, cluster_threshold=4)

        if not cluster_table.empty:
            x = cluster_table.iloc[0]['X']
            y = cluster_table.iloc[0]['Y']
            z = cluster_table.iloc[0]['Z']

            result = ro.r(f'mni_to_region_name(x = {x}, y = {y}, z = {z})')
            parsed = parse_mni_to_region_name(result)

            com_aal_label = parsed['aal']['label']
            com_aal_distance = parsed['aal']['distance']
            com_ba_label = parsed['ba']['label']
            com_ba_distance = parsed['ba']['distance']

            cluster_id_numeric = int(label)
            rows_to_update = table['ClusterID_numeric'] == cluster_id_numeric

            table.loc[rows_to_update, 'Center of Mass X'] = x
            table.loc[rows_to_update, 'Center of Mass Y'] = y
            table.loc[rows_to_update, 'Center of Mass Z'] = z
            table.loc[rows_to_update, 'Center of Mass AAL Label'] = com_aal_label
            table.loc[rows_to_update, 'Center of Mass AAL Distance'] = com_aal_distance
            table.loc[rows_to_update, 'Center of Mass BA Label'] = com_ba_label
            table.loc[rows_to_update, 'Center of Mass BA Distance'] = com_ba_distance
        else:
            #print(f"No clusters found in region for label {label}")
            continue

        region_info = table[table['ClusterID_numeric'] == int(label)]
        if not region_info.empty:
            com_aal_label = region_info.iloc[0]["Center of Mass AAL Label"]
            output_dir = result_dir / "cluster_masks"
            output_dir.mkdir(parents=True, exist_ok=True)
            aal_label_clean = re.sub(r'[^a-zA-Z0-9_]', '_', com_aal_label)
            output_filename = output_dir / f"significant_aal_{aal_label_clean}_label_{int(label)}.nii.gz"
            region_img.to_filename(str(output_filename))
            #print(f"Saved region {com_aal_label} (label {label}) to {output_filename}")

            peak_coords = (region_info.iloc[0]['X'], region_info.iloc[0]['Y'], region_info.iloc[0]['Z'])
            com_coords = (region_info.iloc[0]['Center of Mass X'], region_info.iloc[0]['Center of Mass Y'], region_info.iloc[0]['Center of Mass Z'])

            peak_sphere_mask_data = create_sphere_mask(peak_coords, radius=6, shape=label_img.shape[:3], affine=label_img.affine)
            com_sphere_mask_data = create_sphere_mask(com_coords, radius=6, shape=label_img.shape[:3], affine=label_img.affine)

            peak_sphere_mask_img = image.new_img_like(label_img, peak_sphere_mask_data)
            com_sphere_mask_img = image.new_img_like(label_img, com_sphere_mask_data)

            peak_output_filename = output_dir / f"peak_stat_sphere_aal_{aal_label_clean}_label_{int(label)}.nii.gz"
            peak_sphere_mask_img.to_filename(str(peak_output_filename))
            #print(f"Saved peak stat sphere mask for region {com_aal_label} (label {label}) to {peak_output_filename}")

            com_output_filename = output_dir / f"center_of_mass_sphere_aal_{aal_label_clean}_label_{int(label)}.nii.gz"
            com_sphere_mask_img.to_filename(str(com_output_filename))
            #print(f"Saved center of mass sphere mask for region {com_aal_label} (label {label}) to {com_output_filename}")
        else:
            print(f"No region info found for label {label}")
            
    cluster_report_filename = result_dir / f"{roi}_cluster_table.csv"
    table.to_csv(cluster_report_filename, index=False)
    #print(f"Updated cluster report saved to {cluster_report_filename}")

def prepare_z_maps(z_map_parent_path):
    z_maps_all = {}
    for roi_path in z_map_parent_path.iterdir():
        if roi_path.is_dir():
            roi = roi_path.name
            z_maps_all[roi] = {}

            for z_map_file in roi_path.glob("sub-*_fisher_z_img.nii.gz"):
                subject_id = z_map_file.stem.split('_')[0][4:]  
                z_maps_all[roi][subject_id] = str(z_map_file)

    return z_maps_all

def main():
    ro.r('library(label4MRI)')

    parser = argparse.ArgumentParser(description='Run CWAS analysis.')
    parser.add_argument('--regressor_file', type=str, required=True, help='Regressor file name')
    parser.add_argument('--smoothness', type=str, required=True, help='Smoothness of preprocessing')
    parser.add_argument('--mode', type=str, required=False, help='Mode of operation, e.g., regress', default="regress")
    args = parser.parse_args()

    mdmr_dir = "/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/"
    nas_dir = Path("/mnt/NAS2-2/data/")
    seed_anal_dir = nas_dir / "SAD_gangnam_seed_based_analysis"
    regressor_dir = Path(mdmr_dir) / "regressor"

    data_handler = DataHandler()
    group_num = data_handler.get_subject_group(args.regressor_file)
    # Load the variable name from the specified --regressor_file
    source_variable_name = data_handler.get_variable(args.regressor_file)

    regressor_name = args.regressor_file.replace("_regressor.csv", "")
    regressor_df = pd.read_csv(f"{regressor_dir}/{args.regressor_file}")
    subjects_label = regressor_df["Participant"].values

    z_map_parent_path = seed_anal_dir / f"{args.smoothness}mm" / "corr_z-map" / str(data_handler.get_subject_group(args.regressor_file)) / source_variable_name
    roi_list = [d.name for d in z_map_parent_path.iterdir() if d.is_dir()]
    z_maps_all = prepare_z_maps(z_map_parent_path)

    # Check the mode
    if args.mode == 'regress':
        # Loop through all regressor files and only include those with the same variable name
        all_regressor_files = list(regressor_dir.glob("*_regressor.csv"))
        matching_regressor_files = []
        
        for other_regressor_file_path in all_regressor_files:
            # Extract the variable name from each regressor file
            other_variable_name = data_handler.get_variable(other_regressor_file_path.name)
            
            # Only include files with the same variable name
            if other_variable_name == source_variable_name:
                matching_regressor_files.append(other_regressor_file_path)

        # Replace all_regressor_files with the filtered list
        all_regressor_files = matching_regressor_files

    else:
        # Original behavior: Use all available regressor files
        all_regressor_files = list(regressor_dir.glob("*_regressor.csv"))
        all_regressor_files = [file for file in all_regressor_files if "all_pseudo_variable_regressor.csv" not in file.name]

    for other_regressor_file_path in tqdm(all_regressor_files):
        other_regressor_file = other_regressor_file_path.name
        #print(f"Processing second-level analysis with regressor file: {other_regressor_file}")

        other_variable_name = data_handler.get_variable(other_regressor_file)
        other_regressor_name = other_regressor_file.replace("_regressor.csv", "")
        other_regressor_df = pd.read_csv(other_regressor_file_path)
        other_subjects_label = other_regressor_df["Participant"].values

        extra_info_subjects = pd.DataFrame({
            "subject_label": other_subjects_label,
            other_variable_name: other_regressor_df[other_variable_name],
            "sex": other_regressor_df["SEX"],
            "age": other_regressor_df["AGE"],
            "yr_edu": other_regressor_df["YR_EDU"],
            "mean_framewise_displacement": other_regressor_df["Mean_Framewise_Displacement"]
        })

        design_matrix = make_second_level_design_matrix(
            other_subjects_label, extra_info_subjects
        )

        for roi in roi_list:
            #print(f"Processing ROI: {roi}")
            z_maps = []
            for subject_id in other_subjects_label:
                if subject_id in z_maps_all[roi]:
                    z_maps.append(z_maps_all[roi][subject_id])
                else:
                    print(f"Z-map for subject {subject_id} not found in ROI {roi}. Skipping subject.")
            if not z_maps:
                print(f"No z-maps found for ROI {roi} and regressor {other_regressor_name}. Skipping ROI.")
                continue
            process_roi(roi, group_num, source_variable_name, other_variable_name, design_matrix, other_regressor_df, args.smoothness, other_regressor_name, z_maps, other_regressor_file_path)
def add_region_labels(cluster_table):
    aal_labels = []
    aal_distances = []
    ba_labels = []
    ba_distances = []

    for _, row in cluster_table.iterrows():
        x, y, z = row['X'], row['Y'], row['Z']
        result = ro.r(f'mni_to_region_name(x = {x}, y = {y}, z = {z})')
        parsed = parse_mni_to_region_name(result)

        # AAL 정보
        aal_labels.append(parsed['aal']['label'])
        aal_distances.append(parsed['aal']['distance'])

        # BA 정보
        ba_labels.append(parsed['ba']['label'])
        ba_distances.append(parsed['ba']['distance'])

    # 테이블에 레이블 추가
    cluster_table["AAL Label"] = aal_labels
    cluster_table["AAL Distance"] = aal_distances
    cluster_table["BA Label"] = ba_labels
    cluster_table["BA Distance"] = ba_distances

    return cluster_table
def process_roi(roi, group_num, source_variable_name, other_variable_name, design_matrix, regressor_df, smoothness, regressor_name, z_maps, other_regressor_file_path):
    from nilearn import image
    from nilearn.glm.second_level import SecondLevelModel

    second_level_model = SecondLevelModel(n_jobs=-1)
    second_level_model = second_level_model.fit(z_maps, design_matrix=design_matrix)

    p_map = second_level_model.compute_contrast(other_variable_name, output_type="p_value")
    t_map = second_level_model.compute_contrast(other_variable_name, output_type="stat")

    result_dir = Path("/mnt/NAS2-2/data/") / "SAD_gangnam_seed_based_analysis" / f"{smoothness}mm" / "second_level_results" / str(group_num) / source_variable_name / roi / other_regressor_file_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)

    p_map.to_filename(result_dir / f"{roi}_p_map.nii.gz")
    t_map.to_filename(result_dir / f"{roi}_t_map.nii.gz")

    mask_img = image.math_img('img < 0.05', img=p_map)
    masked_t_map = image.math_img('img1 * img2', img1=t_map, img2=mask_img)

    t_data = masked_t_map.get_fdata()
    t_data_nonzero = t_data[t_data != 0]
    if t_data_nonzero.size == 0:
        #print(f"No significant clusters found for ROI {roi} and regressor {regressor_name}.")
        return
    t_min = np.min(t_data_nonzero)

    cluster_table, label_maps = get_clusters_table(
        masked_t_map, stat_threshold=t_min, cluster_threshold=4, return_label_maps=True
    )

    if cluster_table.empty:
        #print(f"No clusters found for ROI {roi} and regressor {regressor_name}.")
        return

    cluster_table = add_region_labels(cluster_table)

    extract_cluster_mask(
        group=group_num,
        variable=source_variable_name,
        smoothness=smoothness,
        roi=roi,
        table=cluster_table,
        label_maps=label_maps,
        mask_img=mask_img,
        regressor_name=regressor_name,
        other_regressor_file_path=other_regressor_file_path,
        other_variable_name=other_variable_name
    )

if __name__ == "__main__":
    main()
