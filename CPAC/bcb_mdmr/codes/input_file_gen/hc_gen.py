import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def main(group_name, predictor_file_name, variable_of_interest):
    # Load the data
    data = pd.read_csv(predictor_file_name)
    
    # Load valid participants from the CSV file and remove 'sub-' prefix
    valid_participants = pd.read_csv("../../input/valid_after_post_fmriprep_processing.csv")['Participant']
    valid_participants = valid_participants.str.replace('sub-', '', regex=False)

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
    
    # Only keep participants present in the valid participant list
    data_filtered = data_filtered[data_filtered['Participant'].isin(valid_participants)]
    
    # Filter participants whose IDs start with 's'
    data_filtered = data_filtered[data_filtered['Participant'].str.startswith('c')]

    if variable_of_interest in data.columns:
        print(f"Processing {variable_of_interest}")
        
        # Select relevant columns for the regressor file
        regressor_data = data_filtered[['Participant', 'SEX', 'AGE', 'YR_EDU', variable_of_interest]].copy()
    
        # Create a list to store the mean framewise displacement for each participant
        mean_displacement_list = []

        participant_list = regressor_data['Participant'].tolist()

        for participant in participant_list:
            mean = mean_framewise_displacement(participant)
            mean_displacement_list.append(mean)

        # Add the mean framewise displacement to the regressor data
        regressor_data.loc[:, 'Mean_Framewise_Displacement'] = mean_displacement_list

        # Save the original regressor file (non-scaled)
        regressor_file_name = f"../../regressor/{group_name}_{variable_of_interest}_regressor_non_scaled.csv"
        regressor_data.to_csv(regressor_file_name, index=False)
        print(f"Non-scaled regressor file saved as {regressor_file_name}")

        # Standardize (scale) the regressor data, excluding SEX
        scaler = StandardScaler()
        regressor_data_scaled = regressor_data.copy()
        regressor_data_scaled[['AGE', 'YR_EDU', variable_of_interest, 'Mean_Framewise_Displacement']] = scaler.fit_transform(
            regressor_data[['AGE', 'YR_EDU', variable_of_interest, 'Mean_Framewise_Displacement']]
        )

        # Save the scaled regressor file
        regressor_file_name_scaled = f"../../regressor/{group_name}_{variable_of_interest}_regressor_scaled.csv"
        regressor_data_scaled.to_csv(regressor_file_name_scaled, index=False)
        print(f"Scaled regressor file saved as {regressor_file_name_scaled}")

        # (Other transformations remain unchanged)

    else:
        print(f"Variable {variable_of_interest} not found in the dataset.")
        
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
    variable_list = ["STAI-X-1","STAI-X-2","HADS_anxiety","HADS_depression","SWLS","GAD-7","PDSS","LSAS_performance","LSAS_social_interaction","LSAS","MOCI","MOCI_checking","MOCI_cleaning","MOCI_doubting","MOCI_slowness","BFNE","PSWQ","FCV-19S","LSAS_performance_fear","LSAS_performance_avoidance","LSAS_social_fear","LSAS_social_avoidance","LSAS_fear","LSAS_avoidance"]
    group_name = "gangnam_hc"
    predictor_file_name = "../../input/participant_demo_clinical_all_new.csv"
    for variable in variable_list:
        main(group_name, predictor_file_name, variable)
