import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_diagnosis_regressor_with_covariates(group_name, predictor_file_name):
    # Load the data
    data = pd.read_csv(predictor_file_name)
    
    # Load valid participants from the CSV file and remove 'sub-' prefix
    valid_participants = pd.read_csv("../../input/valid_after_post_fmriprep_processing.csv")['Participant']
    valid_participants = valid_participants.str.replace('sub-', '', regex=False)

    # Assume 'fmri_code', '1. SEX', and '2.AGE' are the columns in the dataset
    current_participant_column = 'fmri_code'  # Replace with actual column name
    current_sex_column = '1. SEX'  # Replace with actual column name
    current_age_column = '2.AGE'   # Replace with actual column name
    
    # Rename the columns to 'Participant', 'SEX', and 'AGE'
    data = data.rename(columns={
        current_participant_column: 'Participant',
        current_sex_column: 'SEX',
        current_age_column: 'AGE'
    })
    
    # Convert SEX: 1 for male, 2 for female to 1 for male, 0 for female
    data['SEX'] = data['SEX'].map({1: 1, 2: 0})
    
    # Only keep participants present in the valid participant list
    data_filtered = data[data['Participant'].isin(valid_participants)]

    # Filter out rows where any of the specified columns are missing
    required_columns = ['LSAS', 'LSAS_performance', 'LSAS_social_interaction', 'SEX', 'AGE']
    data_filtered = data_filtered.dropna(subset=required_columns)

    # Create a diagnosis regressor where 's' starts with 1 and 'c' starts with -1
    data_filtered['Diagnosis'] = data_filtered['Participant'].apply(
        lambda x: 1 if x.startswith('s') else (-1 if x.startswith('c') else 0)
    )

    # Create a list to store the mean framewise displacement for each participant
    mean_displacement_list = []
    participant_list = data_filtered['Participant'].tolist()

    for participant in participant_list:
        mean = mean_framewise_displacement(participant)
        mean_displacement_list.append(mean)

    # Add the mean framewise displacement to the regressor data
    data_filtered['Mean_Framewise_Displacement'] = mean_displacement_list

    # Drop rows with missing values in 'SEX', 'AGE', and 'Mean_Framewise_Displacement'
    data_filtered = data_filtered.dropna(subset=['SEX', 'AGE', 'Mean_Framewise_Displacement'])

    """# Standardize (scale) the regressor data, excluding SEX
    scaler = StandardScaler()
    data_filtered[['AGE', 'Mean_Framewise_Displacement']] = scaler.fit_transform(
        data_filtered[['AGE', 'Mean_Framewise_Displacement']]
    )"""
    
    # Create the regressor file with Participant, SEX, AGE, Mean_Framewise_Displacement, and Diagnosis
    regressor_file_name = f"../../regressor/{group_name}_diagnosis_regressor_with_covariates.csv"
    data_filtered[['Participant', 'SEX', 'AGE', 'Mean_Framewise_Displacement', 'Diagnosis']].to_csv(regressor_file_name, index=False)
    print(f"Diagnosis regressor file saved as {regressor_file_name}")

def mean_framewise_displacement(participant):
    try:
        confounds_file = f"/mnt/NAS2-2/data/SAD_gangnam_resting_2/fMRIPrep_total/sub-{participant}/ses-01/func/sub-{participant}_ses-01_task-rest_desc-confounds_timeseries.tsv"
        confound = pd.read_csv(confounds_file, delimiter='\t')
        # Exclude the first row and calculate the mean of the remaining values
        mean = confound['framewise_displacement'].iloc[1:].mean()
        return mean
    except:
        print(f"Participant {participant} fMRI data not exists")
        return -1

if __name__ == "__main__":
    group_name = "gangnam_total"
    predictor_file_name = "../../input/participant_demo_clinical_all_new.csv"
    
    # Create diagnosis regressor with covariates
    create_diagnosis_regressor_with_covariates(group_name, predictor_file_name)
