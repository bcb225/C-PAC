import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from nilearn import datasets, image
import os

# 1. Load Yeo 17-network and 7-network atlases
atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas_yeo_17 = atlas_yeo_2011.thick_17  # Use 17-network atlas
atlas_yeo_7 = atlas_yeo_2011.thick_7    # Use 7-network atlas

# Yeo 17 network names
yeo_17_network_names = {
    1: "Visual_Central_Visual_A",
    2: "Visual_Peripheral_Visual_B",
    3: "Somatomotor_A",
    4: "Somatomotor_B",
    5: "Dorsal_Attention_A",
    6: "Dorsal_Attention_B",
    7: "Salience_Ventral_Attention_A",
    8: "Salience_Ventral_Attention_B",
    9: "Limbic_A",
    10: "Limbic_B",
    11: "Control_C",
    12: "Control_A",
    13: "Control_B",
    14: "Temporal_Parietal",
    15: "Default_C",
    16: "Default_A",
    17: "Default_B"
}

# Yeo 7 network names
yeo_7_network_names = {
    1: "Visual_Network_VIS",
    2: "Somatomotor_Network_SMN",
    3: "Dorsal_Attention_Network_DAN",
    4: "Salience_Ventral_Attention_Network_VAN",
    5: "Limbic_Network_LIM",
    6: "Control_Network_CON",
    7: "Default_Mode_Network_DMN"
}

# 2. Load the cluster image (input your cluster file path)
cluster_img = nib.load('/mnt/NAS2-2/data/SAD_gangnam_MDMR/6mm/3/LSAS_avoidance/result/cluster_masks/MDMR_significant_aal(Frontal_Mid_R)_label(1).nii.gz')
cluster_data = cluster_img.get_fdata()

# 3. Resample the Yeo 17 and 7 atlases to match the cluster image resolution
resampled_atlas_17 = image.resample_to_img(atlas_yeo_17, cluster_img, interpolation='nearest')
resampled_atlas_7 = image.resample_to_img(atlas_yeo_7, cluster_img, interpolation='nearest')
resampled_17_data = np.squeeze(resampled_atlas_17.get_fdata())  # Remove singleton dimensions
resampled_7_data = np.squeeze(resampled_atlas_7.get_fdata())    # Remove singleton dimensions

# 4. Output directory to save individual network clusters (ensure these directories exist)
output_dir_17 = './yeo_17_clusters/'
output_dir_7 = './yeo_7_clusters/'
os.makedirs(output_dir_17, exist_ok=True)
os.makedirs(output_dir_7, exist_ok=True)

# 5. Save Yeo 17 network clusters with names
for i in range(1, 18):  # Yeo 17 network labels are from 1 to 17
    # Create a mask for the current network (i.e., select voxels belonging to network i)
    network_mask = (resampled_17_data == i)
    
    # Apply the mask to the cluster data (retain only voxels that belong to the network and cluster)
    network_cluster = network_mask.astype(int)  # Convert to integer type (1 for network, 0 for others)
    
    # Create a new NIfTI image with the same affine and header as the cluster image
    network_cluster_img = nib.Nifti1Image(network_cluster, affine=cluster_img.affine, header=cluster_img.header)
    
    # Define the output file path with network name
    network_name = yeo_17_network_names[i]
    output_path = f'{output_dir_17}yeo17_network_{i}_{network_name}_cluster.nii.gz'
    
    # Save the NIfTI image
    nib.save(network_cluster_img, output_path)
    print(f"Saved Yeo 17 network {i} ({network_name}) cluster to {output_path}")

# 6. Save Yeo 7 network clusters with names
for i in range(1, 8):  # Yeo 7 network labels are from 1 to 7
    # Create a mask for the current network (i.e., select voxels belonging to network i)
    network_mask = (resampled_7_data == i)
    
    # Apply the mask to the cluster data (retain only voxels that belong to the network and cluster)
    network_cluster = network_mask.astype(int)  # Convert to integer type (1 for network, 0 for others)
    
    # Create a new NIfTI image with the same affine and header as the cluster image
    network_cluster_img = nib.Nifti1Image(network_cluster, affine=cluster_img.affine, header=cluster_img.header)
    
    # Define the output file path with network name
    network_name = yeo_7_network_names[i]
    output_path = f'{output_dir_7}yeo7_network_{i}_{network_name}_cluster.nii.gz'
    
    # Save the NIfTI image
    nib.save(network_cluster_img, output_path)
    print(f"Saved Yeo 7 network {i} ({network_name}) cluster to {output_path}")
