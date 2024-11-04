import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from nilearn import datasets, image
import os

# 1. Load Yeo 7-network atlas
atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas_yeo_7 = atlas_yeo_2011.thick_7  # Use 7-network atlas

# 2. Load the cluster image (input your cluster file path)
cluster_img = nib.load('/mnt/NAS2-2/data/SAD_gangnam_MDMR/6mm/3/LSAS_avoidance/result/cluster_masks/MDMR_significant_aal(Frontal_Mid_R)_label(1).nii.gz')
cluster_data = cluster_img.get_fdata()

# 3. Resample the Yeo 7 atlas to match the cluster image resolution
resampled_atlas_7 = image.resample_to_img(atlas_yeo_7, cluster_img, interpolation='nearest')
resampled_7_data = np.squeeze(resampled_atlas_7.get_fdata())  # Remove singleton dimensions

# 4. Output directory to save individual network clusters (ensure this directory exists, or create it)
output_dir_7 = './yeo_7_clusters/'  # Define your output directory
os.makedirs(output_dir_7, exist_ok=True)  # Create the directory if it doesn't exist

# 5. Iterate through the Yeo 7 network labels (1 to 7) and save each cluster as a NIfTI file
for i in range(1, 8):  # Yeo 7 network labels are from 1 to 7
    # Create a mask for the current network (i.e., select voxels belonging to network i)
    network_mask = (resampled_7_data == i)
    
    # Apply the mask to the cluster data (retain only voxels that belong to the network and cluster)
    network_cluster = network_mask.astype(int)  # Convert to integer type (1 for network, 0 for others)
    
    # Create a new NIfTI image with the same affine and header as the cluster image
    network_cluster_img = nib.Nifti1Image(network_cluster, affine=cluster_img.affine, header=cluster_img.header)
    
    # Define the output file path
    output_path = f'{output_dir_7}yeo7_network_{i}_cluster.nii.gz'
    
    # Save the NIfTI image
    nib.save(network_cluster_img, output_path)
    print(f"Saved Yeo 7 network {i} cluster to {output_path}")
