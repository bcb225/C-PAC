group_list = ["gangnam_total", "gangnam_sad"]
variable_list = ["LSAS","LSAS_performance","LSAS_social_interaction","LSAS_performance_fear","LSAS_performance_avoidance","LSAS_social_fear","LSAS_social_avoidance","LSAS_fear","LSAS_avoidance", "BFNE_S","BFNE", "HADS_anxiety","HADS_depression", "MOCI","MOCI_checking","MOCI_cleaning","MOCI_doubting","MOCI_slowness","STAI-X-1","STAI-X-2","SWLS","GAD-7","PDSS","PSWQ","FCV-19S"]
for smoothness in ['8']:
    for variable in variable_list:
        for group in group_list:
            command = f"python run_mdmr.py --smoothness {smoothness} --mode scan --regressor_file {group}_{variable}_regressor_non_scaled.csv &&"
            print(command)