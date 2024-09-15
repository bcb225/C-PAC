import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def process_report(i, group, variable, smoothness):
    report_path = f"/mnt/NAS2-2/data/SAD_gangnam_MDMR/{smoothness}mm/{group}/{variable}/temp/cluster_report/cluster_report_{i}.txt"
    try:
        report = pd.read_csv(report_path, delimiter='\t')
        if 'Voxels' in report.columns and len(report) > 0:
            max_voxel_count = report['Voxels'].values[0]
        else:
            print(f"Warning: File {report_path} is empty or does not contain 'Voxels' column.")
            max_voxel_count = None
    except Exception as e:
        print(f"Error reading {report_path}: {e}")
        max_voxel_count = None
    
    return max_voxel_count

def calculate_threshold_cluster_size(group, variable, smoothness, permutations=15000, alpha_level=0.05, num_cores=20):
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.starmap(process_report, [(i, group, variable, smoothness) for i in range(1, permutations)]),
                            total=permutations-1))

    # None 값 제거
    results = [result for result in results if result is not None]

    if len(results) == 0:
        print("No valid data found.")
        return None

    # numpy array로 변환
    max_voxel_list = np.array(results)
    print("Max Cluster Sizes Distribution:", max_voxel_list)

    # 리스트 정렬
    sorted_voxel_list = np.sort(max_voxel_list)

    # 알파 레벨에 해당하는 인덱스 계산
    index = int(np.floor(len(sorted_voxel_list) * (1 - alpha_level)))

    # 해당 인덱스의 클러스터 크기 추출
    threshold_cluster_size = sorted_voxel_list[index]

    print(f"Alpha level of {alpha_level*100}% cluster size: {threshold_cluster_size}")
    return threshold_cluster_size

if __name__ == "__main__":
    # argparse 설정
    parser = argparse.ArgumentParser(description="Calculate threshold cluster size.")
    parser.add_argument("--group", required=True, help="Group name (e.g., 'gangnam_total')")
    parser.add_argument("--variable", required=True, help="Variable name (e.g., 'LSAS')")
    parser.add_argument("--smoothness", required=True, help="Smoothness 6 or 8")
    parser.add_argument("--permutations", type=int, default=15000, help="Number of permutations (default: 15000)")
    parser.add_argument("--alpha_level", type=float, default=0.05, help="Alpha level (default: 0.05)")
    parser.add_argument("--num_cores", type=int, default=20, help="Number of cores to use (default: 20)")

    args = parser.parse_args()

    # 인자에 따라 함수 호출
    threshold_cluster_size = calculate_threshold_cluster_size(
        args.group, args.variable, args.smoothness, args.permutations, args.alpha_level, args.num_cores
    )
