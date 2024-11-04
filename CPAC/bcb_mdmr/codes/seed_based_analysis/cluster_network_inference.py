import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np
from nilearn import datasets
from nilearn import image

# 1. Load Yeo 17-network and 7-network atlases
atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas_yeo_17 = atlas_yeo_2011.thick_17  # Use 17-network atlas
atlas_yeo_7 = atlas_yeo_2011.thick_7    # Use 7-network atlas

yeo_17_network_names = {
    1: "Visual Central (Visual A)",               # Visual Central
    2: "Visual Peripheral (Visual B)",            # Visual Peripheral
    3: "Somatomotor A",                           # Somatomotor A
    4: "Somatomotor B",                           # Somatomotor B
    5: "Dorsal Attention A",                      # Dorsal Attention A
    6: "Dorsal Attention B",                      # Dorsal Attention B
    7: "Salience / Ventral Attention A",          # Salience / Ventral Attention A
    8: "Salience / Ventral Attention B",          # Salience / Ventral Attention B
    9: "Limbic A",                                # Limbic A
    10: "Limbic B",                               # Limbic B
    11: "Control C",                              # Control C
    12: "Control A",                              # Control A
    13: "Control B",                              # Control B
    14: "Temporal Parietal",                      # Temporal Parietal Network
    15: "Default C",                              # Default Mode Network C
    16: "Default A",                              # Default Mode Network A
    17: "Default B"                               # Default Mode Network B
}

yeo_7_network_names = {
    1: "Visual Network (VIS)",                # Visual Network
    2: "Somatomotor Network (SMN)",           # Somatomotor Network
    3: "Dorsal Attention Network (DAN)",      # Dorsal Attention Network
    4: "Salience / Ventral Attention Network (VAN)",  # Salience / Ventral Attention Network
    5: "Limbic Network (LIM)",                # Limbic Network
    6: "Control Network (CON)",               # Control Network (Frontoparietal)
    7: "Default Mode Network (DMN)"           # Default Mode Network
}

# 2. Load cluster NIfTI file (input your cluster file path)
cluster_img = nib.load('/mnt/NAS2-2/data/SAD_gangnam_MDMR/6mm/3/LSAS_avoidance/result/cluster_masks/MDMR_significant_aal(Frontal_Mid_R)_label(1).nii.gz')
cluster_data = cluster_img.get_fdata()

# 3. Resample Yeo atlases to match the cluster image resolution
resampled_atlas_17 = image.resample_to_img(atlas_yeo_17, cluster_img, interpolation='nearest')
resampled_atlas_7 = image.resample_to_img(atlas_yeo_7, cluster_img, interpolation='nearest')

# 4. Find voxels labeled as 1 within the cluster
cluster_voxels = (cluster_data == 1)

# 5. Reshape resampled data to match cluster image dimensions
resampled_17_data = np.squeeze(resampled_atlas_17.get_fdata())  # Remove singleton dimensions
resampled_7_data = np.squeeze(resampled_atlas_7.get_fdata())    # Remove singleton dimensions

# 6. Calculate the total number of cluster voxels
total_cluster_voxels = np.sum(cluster_voxels)

# 7. Find overlapping regions between the cluster and Yeo 17 networks
overlap_17 = {i: 0 for i in range(1, 18)}

# Iterate through the Yeo 17 network labels to find overlap with the cluster voxels
for i in range(1, 18):  # Yeo 17 network labels are from 1 to 17
    overlap_voxels = np.logical_and(resampled_17_data == i, cluster_voxels)
    overlap_17[i] = np.sum(overlap_voxels)

# Print the results showing the number of overlapping voxels and the percentage for each network
print("Yeo 17-network overlap:")
for i, count in overlap_17.items():
    if count > 0:
        percentage = (count / total_cluster_voxels) * 100
        print(f"Overlap with {yeo_17_network_names[i]}: {count} voxels ({percentage:.2f}%)")

# 8. Find overlapping regions between the cluster and Yeo 7 networks
overlap_7 = {i: 0 for i in range(1, 8)}

# Iterate through the Yeo 7 network labels to find overlap with the cluster voxels
for i in range(1, 8):  # Yeo 7 network labels are from 1 to 7
    overlap_voxels = np.logical_and(resampled_7_data == i, cluster_voxels)
    overlap_7[i] = np.sum(overlap_voxels)

# Print the results showing the number of overlapping voxels and the percentage for each network
print("\nYeo 7-network overlap:")
for i, count in overlap_7.items():
    if count > 0:
        percentage = (count / total_cluster_voxels) * 100
        print(f"Overlap with {yeo_7_network_names[i]}: {count} voxels ({percentage:.2f}%)")
