import pandas as pd
import numpy as np

# Load the valid participants from the CSV file and remove 'sub-' prefix
valid_participants_df = pd.read_csv("../../input/valid_after_post_fmriprep_processing.csv")
valid_participants = valid_participants_df['Participant'].str.replace('sub-', '', regex=False).tolist()

# Create the pseudo_variable and pseudo_covariates values randomly
pseudo_variable = np.random.rand(len(valid_participants))
pseudo_covariates = np.random.rand(len(valid_participants))

# Create the DataFrame for all_pseudo_variable_regressor.csv
pseudo_regressor_df = pd.DataFrame({
    'Participant': valid_participants,
    'pseudo_variable': pseudo_variable,
    'pseudo_covariates': pseudo_covariates
})

# Save the DataFrame to all_pseudo_variable_regressor.csv
pseudo_regressor_df.to_csv('../../regressor/all_pseudo_variable_regressor.csv', index=False)
print("all_pseudo_variable_regressor.csv has been created.")

# Save the valid participants to all_code_list.csv without a header
pd.Series(valid_participants).to_csv('../../regressor/all_code_list.csv', index=False, header=False)
print("all_code_list.csv has been created.")
