import pandas as pd
import os
import argparse
from pathlib import Path

def main(group_name, predictor_file_name, dp_type, week_type):
    # Load the data from the HC and SAD files
    mnt_dir = "/mnt/NAS2-2/data/"
    dp_dir = f"{mnt_dir}/SAD_gangnam_DP/"
    feature_dir = f"{dp_dir}/dp_features/"
    feature_path = Path(feature_dir)
    
    sad_data_file = f"{dp_type}_{week_type}_sad_window.csv"
    hc_data_file = f"{dp_type}_{week_type}_hc_window.csv"
    
    # Read the SAD and HC data files
    sad_data = pd.read_csv(feature_path / dp_type / "SAD" / sad_data_file)
    hc_data = pd.read_csv(feature_path / dp_type / "HC" / hc_data_file)
    
    # Drop columns with any missing values
    sad_data = sad_data.dropna(axis=1)
    hc_data = hc_data.dropna(axis=1)
    
    # Filter rows where the 'chunk' column equals 0
    sad_first_chunk = sad_data[sad_data['chunk_num'] == 0]
    hc_first_chunk = hc_data[hc_data['chunk_num'] == 0]
    
    # Combine SAD and HC data into a single dataframe
    combined_data = pd.concat([sad_first_chunk, hc_first_chunk], ignore_index=True)

    # Load the original predictor file (participant demographics)
    data = pd.read_csv(predictor_file_name)
    
    # Rename columns for consistency
    current_sex_column = '1. SEX'
    current_age_column = '2.AGE'
    current_education_column = '3-2. YR_EDU'
    current_participant_column = 'fmri_code'

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
    required_columns = ['SEX', 'AGE', 'YR_EDU']
    data_filtered = data.dropna(subset=required_columns)

    # Process the combined HC and SAD data
    process_group(group_name, data_filtered, combined_data, week_type)

def process_group(group_name, data_filtered, combined_data, week_type):
    # Get the list of participants from the combined data
    participant_list = combined_data['fmri_code'].tolist()
    
    # Filter participants in the predictor file
    data_filtered = data_filtered[data_filtered['Participant'].isin(participant_list)]
    
    # Create a new participant list based on framewise displacement
    new_participant_list = []
    mean_displacement_list = []

    # Dictionary to store the mean framewise displacement for each participant
    displacement_dict = {}

    for participant in data_filtered['Participant']:
        mean = mean_framewise_displacement(participant)
        if mean >= 0:
            new_participant_list.append(participant)
            mean_displacement_list.append(mean)
            displacement_dict[participant] = mean

    # Update data_filtered to only include valid participants
    data_filtered = data_filtered[data_filtered['Participant'].isin(new_participant_list)]
    
    # Add the mean framewise displacement column to data_filtered
    data_filtered['Mean_Framewise_Displacement'] = data_filtered['Participant'].map(displacement_dict)

    # Exclude unnecessary columns for the regressor file
    columns_to_exclude = ['subjNum', 'fmri_code', 'fmri_date', 'window_start_date', 'window_end_date', 'row_number', 'chunk_num']
    combined_data_filtered = combined_data.drop(columns=columns_to_exclude)

    # Set of participants who have valid data for all variables that are saved
    valid_participants = set(data_filtered['Participant'])

    # Iterate over each remaining variable in combined_data
    for variable in combined_data_filtered.columns:
        # Skip variables that start with 'Unnamed'
        if variable.startswith('Unnamed'):
            print(f"Skipping {variable} as it appears to be an index or unnecessary column.")
            continue

        print(f"Processing {variable} for {group_name}")

        # Add the current variable to data_filtered and rename it with week_type
        variable_name_with_week = f"{variable}_{week_type}"
        data_filtered_variable = data_filtered[['Participant', 'SEX', 'AGE', 'YR_EDU', 'Mean_Framewise_Displacement']].copy()
        data_filtered_variable[variable_name_with_week] = combined_data.set_index('fmri_code').loc[data_filtered_variable['Participant'], variable].values

        # Skip this variable if any participant has missing values (NaN)
        if data_filtered_variable[variable_name_with_week].isna().any():
            print(f"Skipping {variable_name_with_week} due to missing values.")
            # Remove participants who don't have valid data for this variable
            valid_participants = valid_participants.intersection(data_filtered_variable.dropna(subset=[variable_name_with_week])['Participant'])
            continue


        # Create the regressor file with the updated naming convention for each variable
        regressor_file_name = f"../../regressor/{group_name}_{variable_name_with_week}_regressor.csv"
        data_filtered_variable.to_csv(regressor_file_name, index=False)
        print(f"Regressor file saved as {regressor_file_name}")

        # Extract participants that have valid data for this variable and save them as a code list
        code_list = pd.Series(data_filtered_variable['Participant'].unique())
        code_list_file_name = f"../../regressor/{group_name}_code_list.csv"
        code_list.to_csv(code_list_file_name, header=False, index=False)
        print(f"Code list for {variable_name_with_week} saved as {code_list_file_name}")

        # Display summary statistics for the current variable
        print(f"Summary statistics for {variable_name_with_week} in {group_name}:")
        print(data_filtered_variable[variable_name_with_week].describe())

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
    except Exception as e:
        print(f"Participant {participant} fMRI data not exists or error: {e}")
        return -1

if __name__ == "__main__":
    predictor_file_name = "../../input/participant_demo_clinical_all.csv"
    
    group_name_list = ["dp_app", "dp_light","dp_screen","dp_location","dp_call"]
    dp_type_list = ["app", "light","screen", "location","call"]
    week_type_list = ["weekdays","weekends"]
    
    for dp_type in dp_type_list:
        for week_type in week_type_list:
            group_name = f"dp_{dp_type}"
            main(group_name, predictor_file_name, dp_type, week_type)
