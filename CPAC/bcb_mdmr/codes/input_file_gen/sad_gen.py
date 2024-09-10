import pandas as pd
import os
import argparse

def main(group_name, predictor_file_name, variable_of_interest):
    # Load the data
    data = pd.read_csv(predictor_file_name)
    
    # Assume these are the current names in the dataset for SEX, AGE, and YR_EDU
    current_sex_column = '1. SEX'  # Replace with the actual column name
    current_age_column = '2.AGE'   # Replace with the actual column name
    current_education_column = '3-2. YR_EDU'  # Replace with the actual column name
    current_participant_column = 'fmri_code'  # Replace with the actual column name

    
    # Rename the columns to SEX, AGE, and YR_EDU
    data = data.rename(columns={
        current_sex_column: 'SEX',
        current_age_column: 'AGE',
        current_education_column: 'YR_EDU',
        current_participant_column: 'Participant'
    })
    
    # Convert SEX column: 1 for male, 2 for female to 1 for male, 0 for female
    data['SEX'] = data['SEX'].map({1: 1, 2: 0})
    
    # Filter out rows where any of the specified columns are missing
    required_columns = ['LSAS', 'LSAS_performance', 'LSAS_social_interaction', 'SEX', 'AGE', 'YR_EDU']
    data_filtered = data.dropna(subset=required_columns)
    # Keep only rows where 'Participant' starts with 's'
    data_filtered = data_filtered[data_filtered['Participant'].str.startswith('s')]

    
    new_participant_list = []
    participant_list = data_filtered['Participant'].tolist()
    for participant in participant_list:
        mean = mean_framewise_displacement(participant)
        if mean >= 0:
            new_participant_list.append(participant)

    participant_list = new_participant_list
    data_filtered = data_filtered[data_filtered['Participant'].isin(new_participant_list)]
    
    if variable_of_interest in data.columns:
        print(f"Processing {variable_of_interest}")
        # Select relevant columns for the regressor file
        regressor_data = data_filtered[['Participant', 'SEX', 'AGE', 'YR_EDU', variable_of_interest]]
    
        # Create a list to store the mean framewise displacement for each participant
        mean_displacement_list = []

        for participant in participant_list:
            mean = mean_framewise_displacement(participant)
            mean_displacement_list.append(mean)

        # Add the mean framewise displacement to the regressor data
        regressor_data['Mean_Framewise_Displacement'] = mean_displacement_list


        # Create the regressor file
        regressor_file_name = f"../../input/{group_name}_{variable_of_interest}_regressor.csv"

        

        regressor_data.to_csv(regressor_file_name, index=False)
        print(f"Regressor file saved as {regressor_file_name}")
        
        # Create the code list file
        code_list = data_filtered['Participant'].apply(lambda x: x.replace('sub-', ''))
        code_list_file_name = f"../../input/{group_name}_code_list.csv"
        code_list.to_csv(code_list_file_name, header=False, index=False)
        print(f"Code list file saved as {code_list_file_name}")
        
        # Display summary statistics for the variable of interest
        print(f"Data for {variable_of_interest}:")
        print(data_filtered[variable_of_interest].describe())
    else:
        print(f"Variable {variable_of_interest} not found in the dataset.")
        
    # mean framewise disaplacement도 covariate로 추가.
    # 예시 경로: /mnt/NAS2-2/data/SAD_gangnam_resting_2/fMRIPrep_total/sub-c0017/ses-01/func/sub-c0017_ses-01_task-rest_desc-confounds_timeseries.tsv
    # 위 TSV에서 framewise_displacement column의 mean 값을 covariate로 추가해야 함.
    # 
    # LSAS, LSAS_performance, LSAS_social_interaction, 1. SEX,2.AGE,3-2. YR_EDU 없는 사람들 제거
def mean_framewise_displacement(participant):
    try:
        confounds_file = f"/mnt/NAS2-2/data/SAD_gangnam_resting_2/fMRIPrep_total/sub-{participant}/ses-01/func/sub-{participant}_ses-01_task-rest_desc-confounds_timeseries.tsv"
        confound = pd.read_csv(
            confounds_file,
            delimiter='\t'
        )
        # Exclude the first row and calculate the mean of the remaining values
        mean = confound['framewise_displacement'].iloc[1:].mean()
        return mean
    except:
        print(f"Participant {participant} fMRI data not exists")
        return -1
if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Process some data.")
    
    #parser.add_argument("--group", type=str, required=True, help="The name of the group")
    #parser.add_argument("--variable", type=str, required=True, help="The variable of interest to analyze")
    variable_list = ["STAI-X-1","STAI-X-2","HADS_anxiety","HADS_depression","SWLS","GAD-7","PDSS","LSAS_performance","LSAS_social_interaction","LSAS","MOCI","MOCI_checking","MOCI_cleaning","MOCI_doubting","MOCI_slowness","BFNE","PSWQ","FCV-19S"]
    #args = parser.parse_args()
    group_name = "gangnam_sad"
    predictor_file_name = "../../input/participant_demo_clinical_all.csv"
    for variable in variable_list:
        main(group_name, predictor_file_name, variable)