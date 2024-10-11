import pandas as pd
import nibabel as nib
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

# 다대다 관계를 위한 조인 테이블 정의
group_subject_table = Table('group_subject_codes', Base.metadata,
    Column('group_id', Integer, ForeignKey('subject_group.group_id')),
    Column('code_id', Integer, ForeignKey('subject_code.code_id'))
)

class RegressorGroup(Base):
    __tablename__ = 'regressor_group'
    id = Column(Integer, primary_key=True, autoincrement=True)
    regressor_file = Column(String, unique=True)  # Ensure each regressor file is unique
    group_id = Column(Integer, ForeignKey('subject_group.group_id'))

    # Relationship to the SubjectGroup
    subject_group = relationship('SubjectGroup', back_populates='regressor_groups')


class SubjectGroup(Base):
    __tablename__ = 'subject_group'
    group_id = Column(Integer, primary_key=True)
    group_name = Column(String)
    subject_codes = relationship('SubjectCode', secondary=group_subject_table, back_populates='groups')
    regressor_groups = relationship('RegressorGroup', back_populates='subject_group')

class SubjectCode(Base):
    __tablename__ = 'subject_code'
    code_id = Column(Integer, primary_key=True)
    subject_code = Column(String, unique=True)
    groups = relationship('SubjectGroup', secondary=group_subject_table, back_populates='subject_codes')


class DataHandler:
    def __init__(self):
        db_path = '/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/db/mdmr.db'
        engine = create_engine(f'sqlite:///{db_path}')  # mdmr.db 파일과 연결
        Session = sessionmaker(bind=engine)
        self.session = Session()  # 세션 생성
        Base.metadata.create_all(engine)
    def subject_dict_maker(self,regressor_file, smoothness):
        regressor_path = f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/regressor/{regressor_file}"
        regressor = pd.read_csv(regressor_path)
        #temp code subject_code_list = pd.read_csv(subject_code_list_file)
        subject_codes = (regressor["Participant"].tolist())
        subject_dict = {}
        for subject_code in subject_codes:
            subject_fmri_dir = f"/mnt/NAS2-2/data/SAD_gangnam_resting_2/fMRIPrep_total/sub-{subject_code}/ses-01/func/sub-{subject_code}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-smoothed{smoothness}mm_resampled4mm_scrbold.nii.gz"
            subject_dict[subject_code] = subject_fmri_dir
        return subject_dict
    def get_mask_file(self, smoothness):
        #진짜 mask, 모든 피험자 같은 마스크 파일 사용
        return f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/template/all_final_group_mask_{smoothness}mm.nii.gz"
        
        #tiny exmaple 연습용
        #return f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/template/{subject_group}_small_final_group_mask.nii.gz"
    def get_regressor_file(self, regressor_file):
        return f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/regressor/{regressor_file}"
    def get_voxel_range(self, smoothness):
        #mask_file = f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/template/{subject_group}_small_final_group_mask.nii.gz"
        #진짜 mask
        mask_file= f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/template/all_final_group_mask_{smoothness}mm.nii.gz"
        # NIfTI 파일 로드
        img = nib.load(mask_file)
        
        # 이미지 데이터 추출
        data = img.get_fdata()
        
        # 값이 1인 복셀의 개수 세기
        voxel_count = np.sum(data == 1)
        print(f"Total Voxel Count: {voxel_count}")
        return np.arange(0, voxel_count)
    def get_subject_group(self, regressor_file):
        # Check if the regressor_file is already linked to a group
        existing_regressor_group = self.session.query(RegressorGroup).filter_by(regressor_file=regressor_file).first()
        
        if existing_regressor_group:
            # If an existing record is found, return the corresponding group_id
            print(f"Found existing regressor-file to group mapping: {existing_regressor_group.regressor_file} -> Group ID: {existing_regressor_group.group_id}")
            return existing_regressor_group.group_id
        
        # Otherwise, proceed to find or create a new subject group
        regressor_path = f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/regressor/{regressor_file}"
        regressor = pd.read_csv(regressor_path)
        subject_codes = regressor["Participant"].values
        group = self.find_existing_group(subject_codes)
        
        if group is None:
            print("Creating a new group.")
            group = self.create_new_group(subject_codes)
        else:
            print(f"Found existing group with ID: {group.group_id}")
        
        # Create a new entry in the RegressorGroup table to map regressor_file to group_id
        new_regressor_group = RegressorGroup(regressor_file=regressor_file, group_id=group.group_id)
        self.session.add(new_regressor_group)
        self.session.commit()
        
        # Return the group_id
        return group.group_id
            
    
    def find_existing_group(self, subject_codes):
        groups = self.session.query(SubjectGroup).all()
        for group in groups:
            group_codes = set(code.subject_code for code in group.subject_codes)
            if group_codes == set(subject_codes):
                return group
        return None
    
    def create_new_group(self, subject_codes):
        new_group = SubjectGroup(group_name='New Group')
        for code in subject_codes:
            existing_code = self.session.query(SubjectCode).filter_by(subject_code=code).first()
            if not existing_code:
                existing_code = SubjectCode(subject_code=code)
            new_group.subject_codes.append(existing_code)
        self.session.add(new_group)
        self.session.commit()
        return new_group
    def get_variable(self,regressor_file):
        regressor_path = f"/home/changbae/fmri_project/C-PAC/CPAC/bcb_mdmr/regressor/{regressor_file}"
        regressor = pd.read_csv(regressor_path)
        exclude_columns = ['Participant', 'SEX', 'AGE', 'YR_EDU', 'Mean_Framewise_Displacement']
        
        # 제외할 컬럼들을 제외한 나머지 컬럼 이름 추출
        remaining_columns = [col for col in regressor.columns if col not in exclude_columns]
        variables_string = ', '.join(remaining_columns)
        # 나머지 컬럼 이름 반환
        return variables_string