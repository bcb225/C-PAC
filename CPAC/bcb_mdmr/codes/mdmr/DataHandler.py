import pandas as pd
import nibabel as nib
import numpy as np
class DataHandler:
    def __init__(self):
        pass
    def subject_dict_maker(self,subject_group, smoothness):
        subject_code_list_file = f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/input/{subject_group}_code_list.csv"
        subject_code_list = pd.read_csv(subject_code_list_file, header=None)
        subject_codes = (subject_code_list.values)
        subject_dict = {}
        for subject_code in subject_codes:
            subject_code = str(subject_code[0])
            subject_fmri_dir = f"/mnt/NAS2-2/data/SAD_gangnam_resting_2/fMRIPrep_total/sub-{subject_code}/ses-01/func/sub-{subject_code}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed{smoothness}mm_resampled4mm_bold.nii.gz"
            subject_dict[subject_code] = subject_fmri_dir
        return subject_dict
    def get_mask_file(self,subject_group, smoothness):
        #진짜 mask, 모든 피험자 같은 마스크 파일 사용
        return f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/template/gangnam_total_final_group_mask_{smoothness}mm.nii.gz"
        
        #tiny exmaple 연습용
        #return f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/template/{subject_group}_small_final_group_mask.nii.gz"
    def get_regressor_file(self, subject_group, target_variable):
        return f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/input/{subject_group}_{target_variable}_regressor.csv"
    def get_voxel_range(self, subject_group, smoothness):
        #mask_file = f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/template/{subject_group}_small_final_group_mask.nii.gz"
        #진짜 mask
        mask_file= f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/template/gangnam_total_final_group_mask_{smoothness}mm.nii.gz"
        # NIfTI 파일 로드
        img = nib.load(mask_file)
        
        # 이미지 데이터 추출
        data = img.get_fdata()
        
        # 값이 1인 복셀의 개수 세기
        voxel_count = np.sum(data == 1)
        print(f"Total Voxel Count: {voxel_count}")
        return np.arange(0, voxel_count)