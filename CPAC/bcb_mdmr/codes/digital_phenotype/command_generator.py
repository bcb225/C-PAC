from pathlib import Path

# Define input directory
input_dir = "../../input/"
input_path = Path(input_dir)

# Initialize a list to store commands
commands = []
dp_group_names = ["dp_app", "dp_location", "dp_light", "dp_screen", "dp_call"]
variable_dict = {group: [] for group in dp_group_names}
# Iterate through files in the directory
for file in input_path.glob("**/*"):
    # Check if it's a file and matches the condition (starts with 'dp', ends with 'regressor', and does not contain 'code_list')
    if file.is_file() and file.name.startswith("dp") and file.name.endswith("regressor.csv") and "code_list" not in file.name:
        # Split the filename into parts
        parts = file.stem.split("_")
        
        # Extract dp_group (e.g., dp_app, dp_screen, etc.)
        dp_group = "_".join(parts[:2])
        
        # Extract variable (everything after dp_group and remove 'regressor' from the end)
        variable = "_".join(parts[2:]).replace("_regressor", "")
        
        # Create the command
        command = f"python run_mdmr.py --group {dp_group} --variable {variable} --smoothness 6 --mode scan"
        
        # Append the command to the list
        commands.append(command)
        
        if dp_group in variable_dict:
            variable_dict[dp_group].append(variable)

# Join the commands with '&&' to create a single output
output_command = " && ".join(commands)

# Output the final command
#print(output_command)
for dp_group_name in dp_group_names:
    print(f"{dp_group_name} # of variables: {len(variable_dict[dp_group_name])}")