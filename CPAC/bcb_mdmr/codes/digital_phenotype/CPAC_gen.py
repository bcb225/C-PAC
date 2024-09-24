import os

# Define the path to the directory containing the files
directory_path = "../../input/"  # 실제 경로로 변경

# List to store the generated commands
commands = []

# Loop through each file in the directory
for file_name in os.listdir(directory_path):
    if file_name.startswith("dp") and "code_list" not in file_name:
        # Extract the group and variable
        group_end_idx = file_name.find("_", 3)  # 첫 번째 'dp_' 뒤의 구분점 인덱스 찾기
        group = file_name[:group_end_idx]  # 'dp_xxx' 추출
        variable = file_name[group_end_idx + 1:].replace(".csv", "").replace("_regressor", "")  # '_regressor' 제거
        
        # Generate the command
        command = f"python run_mdmr.py --group {group} --variable {variable} --smoothness 6 &&"
        commands.append(command)

# Join all commands into one large command string
output = " ".join(commands)

# Output the final command string
print(output)
