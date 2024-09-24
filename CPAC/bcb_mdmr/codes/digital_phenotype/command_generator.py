dp_call_variables = [
    'phone_calls_rapids_incoming_countmostfrequentcontact_weekends',
    'phone_calls_rapids_incoming_count_weekends',
    'phone_calls_rapids_incoming_distinctcontacts_weekends',
    'phone_calls_rapids_incoming_entropyduration_weekends',
    'phone_calls_rapids_incoming_maxduration_weekends',
    'phone_calls_rapids_incoming_meanduration_weekends',
    'phone_calls_rapids_incoming_minduration_weekends',
    'phone_calls_rapids_incoming_modeduration_weekends',
    'phone_calls_rapids_incoming_stdduration_weekends',
    'phone_calls_rapids_incoming_sumduration_weekends',
    'phone_calls_rapids_incoming_timefirstcall_weekends',
    'phone_calls_rapids_incoming_timelastcall_weekends',
    'phone_calls_rapids_missed_countmostfrequentcontact_weekends',
    'phone_calls_rapids_missed_count_weekends',
    'phone_calls_rapids_missed_distinctcontacts_weekends',
    'phone_calls_rapids_missed_timefirstcall_weekends',
    'phone_calls_rapids_missed_timelastcall_weekends',
    'phone_calls_rapids_outgoing_countmostfrequentcontact_weekends',
    'phone_calls_rapids_outgoing_count_weekends',
    'phone_calls_rapids_outgoing_distinctcontacts_weekends',
    'phone_calls_rapids_outgoing_entropyduration_weekends',
    'phone_calls_rapids_outgoing_maxduration_weekends',
    'phone_calls_rapids_outgoing_meanduration_weekends',
    'phone_calls_rapids_outgoing_minduration_weekends',
    'phone_calls_rapids_outgoing_modeduration_weekends',
    'phone_calls_rapids_outgoing_stdduration_weekends',
    'phone_calls_rapids_outgoing_sumduration_weekends',
    'phone_calls_rapids_outgoing_timefirstcall_weekends',
    'phone_calls_rapids_outgoing_timelastcall_weekends'
]


# Generate MDMR commands for dp_app variables
commands = []
for variable in dp_call_variables:
    command = f"python run_mdmr.py --group dp_call --variable {variable} --smoothness 6 && "
    commands.append(command)

# Write the commands to a text file
output_file_path = "./dp_call_mdmr_commands.txt"
with open(output_file_path, "w") as out_file:
    for command in commands:
        out_file.write(command + "\n")