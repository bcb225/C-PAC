import pandas as pd
import numpy as np
from tqdm import tqdm
import ipdb
import math
import os


# n = 47
anx_dict = {'0827jck': 1, 'na0840': 2, 'spwls915': 3, 'bje5409': 5, 'stevenliu': 8, 'jaewon0422': 10, 'psb8387': 13, 'sgprosix': 18, 'hwooon': 19, 'ja9009': 22, 'winter9722': 30, 'juhyeon12': 31, 'iloveppang': 35, 'seogpfus': 39, 'xodwhg': 40, 'dh7335792': 47, 'sje0917': 48, 'best1234': 49, 'lekisek10044': 58, 'valyria': 62, 'inwoo629': 63, 'yeh3033': 64, 'ghldnjscool': 66, 'tjsgn1456': 67, 'zz0688': 68, 'liser1': 69, '1092ks': 70, 'qkralsrl0930': 72, 'skyjinhyuk': 73, 'crs0103': 75, 'kwj8831': 76, 'tmrkem': 77, 'qjarl312': 78, 'dktj1212': 80, '9507ym': 81, 'iiyang': 82, 'gonghojun': 83, 'jeng1995': 85, 'johnpopper': 89, 'ameju7': 92, 'iceginger': 95, 'pjy680444': 101, 'hanul9795': 102, 'wldbsehf': 105, 'qwertyu': 111, 'gws13': 112} 

# n = 54
hc_dict = {'wkddbswns': 4, 'eagle52109': 6, 'claudia1027': 7, 'tjsdud4918': 9, 'acoustic02': 11, 'sun9482': 12, 'najuna0223': 14, 'wkorotk': 15, 'katarina99': 16, 'boeun7320': 17, 'naya0505': 20, 'loststar88': 21, 'appletree': 23, 'lilyjhrim': 24, 'hgpro6': 25, 'ku3369': 26, 'kjrnet': 27, 'inhiscalling': 28, 'rosebean90': 29, 'alsruds99': 32, 'wonsunyang': 33, 'nar00125': 37, 'lemon0910': 38, 'nuri0506': 41, 'dfsfdgt': 42, 'sena6282': 43, 'saint0820': 44, 'nswbae': 45, 'hjzzang': 46, 'rhkrqhdud77': 50, 'dryflowersoo': 56, 'anna05077': 57, 'cholebera': 60, 'ffddss1448': 74, 'lyn6523': 84, 'mmyoung95': 86, 'gkgkoo1234': 87, 'ljs120706': 88, 'dltmdfuf333': 90, 'jskim928': 91, 'donghoon0829': 93, 'sking23': 94, 'crizitj': 96, 'dpfrls12': 97, 'dlguswlqq': 98, 'nick586': 99, 'darkblue37': 100, 'gnala119': 103, 'dongghkdyd': 104, 'sinsy3439': 106, 'vertigoks': 107, 'byj8031': 108, 'nochan1521': 109, 'kcs2078': 110}

# n = 101
total_dict = {**anx_dict, **hc_dict}

anx_mapping = {1: 7, 2: 5, 3: 4, 5: 12, 8: 8, 10: 19, 13: 14, 18: 32, 19: 35, 22: 27, 30: 43, 31: 45, 35: 60, 39: 57, 40: 63, 47: 74, 48: 79, 49: 76, 58: 95, 62: 105, 63: 109, 64: 116, 66: 128, 67: 143, 68: 151, 69: 140, 70: 152, 72: 157, 73: 164, 75: 177, 76: 189, 77: 181, 78: 169, 80: 194, 81: 206, 82: 219, 83: 226, 85: 223, 89: 250, 92: 261, 95: 279, 101: 295, 102: 305, 105: 316, 111: 339, 112: 303}

hc_mapping = {4: 9, 6: 10, 7: 13, 9: 15, 11: 17, 12: 20, 14: 24, 15: 22, 16: 25, 17: 28, 20: 30, 21: 38, 23: 44, 24: 41, 25: 36, 26: 42, 27: 47, 28: 48, 29: 34, 32: 50, 33: 52, 37: 59, 38: 62, 41: 61, 42: 64, 43: 70, 44: 69, 45: 67, 46: 71, 50: 82, 56: 90, 57: 89, 60: 91, 74: 170, 84: 246, 86: 254, 87: 253, 88: 256, 90: 289, 91: 268, 93: 269, 94: 273, 96: 270, 97: 272, 98: 285, 99: 281, 100: 287, 103: 290, 104: 292, 106: 280, 107: 293, 108: 291, 109: 311, 110: 330}

total_mapping = {**anx_mapping, **hc_mapping}



def check_items_in_list(dataframe, check_list):
    """
    Splits the DataFrame into two based on whether each row's first item is in the given list.

    Args:
    dataframe: DataFrame to be checked.
    check_list: List containing items to be checked against the DataFrame.

    Returns:
    Tuple of two DataFrames - the first contains rows with the first item in check_list, the second contains the remaining rows.
    """
    
    result = []
    app_to_add = []
    for _, row in dataframe.iterrows():
        if row[0] in check_list:
            result.append(row)
        else:
            app_to_add.append(row)
            
    return pd.concat(result, axis=1).T, pd.concat(app_to_add, axis=1).T

# Define custom categories for apps (defined by package name)
chat_custom = ['com.facebookorca', 'com.facebook.npe.tuned', 'com.whatsapp', 'org.telegram.messenger', 'com.kakao.talk', 'jp.naver.line.android', 'com.samsung.android.messaging', 'com.sumone', 'com.discord', 'com.bnc.bambi.client', 'kr.co.april7.buddy', 'kr.jungrammer.superranchat', 'com.duriduri.v1', 'com.joyinfo.jubetalk', 'com.dorsia.amanda', 'me.togather.wave', 'com.scatterlab.messenger', 'com.azarlive.android', 'kr.co.vcnc.android.couple', 'land.lifeoasis.maum', 'com.sens.talk', 'kr.co.hiver', 'com.bookpalcomics.secretlove', 'im.thebot.messenger', 'com.tencent.mm', 'com.valvesoftware.android.steam.friendsui', 'com.yangpark.connecting', 'com.probits.argo', 'com.google.android.apps.dynamite', 'com.google.android.apps.messaging', 'Uxpp.UC', 'net.sjang.sail']  # Apps primarily for 1:1 conversation
sns_custom = ['com.facebook.katana', 'com.twitter.android', 'com.instagram.android', 'com.linkedin.android', 'com.google.android.youtube', 'com.vanced.android.youtube', 'com.google.android.apps.youtube.creator', 'com.vimeo.android.videoapp', 'com.flickr.android', 'com.ss.android.ugc.trill', 'com.snapchat.android', 'com.pinterest', 'tv.twitch.android.app', 'com.naver.vapp', 'com.cyworld.minihompy', 'io.cyclub.app', 'com.movefastcompany.bora', 'co.spoonme', 'com.jjaanncompany.jjaann', 'kr.co.nowcom.mobile.afreeca', 'com.imgur.mobile', 'io.celebe', 'com.sec.penup', 'com.hellotalk', 'com.movedot.clpick', 'xyz.thingapps.regram', 'com.noyesrun.meeff.kr', 'com.tape.popsapp', 'com.tumblr', 'co.benx.weverse', 'co.okit.android.app', 'app.revanced.android.youtube', 'app.rvx.android.youtube']  # Apps for communication, not limited to chatting, including posting features
community_custom = ['com.reddit.frontpage', 'net.daum.android.cafe', 'com.towneers.www', 'com.nhn.android.band', 'com.nhn.android.navercafe', 'com.nhn.android.blog', 'com.nate.pann', 'com.rndeep.fns_fantoo', 'kr.munto.app', 'com.frientrip.frip', 'com.teamblind.blind', 'net.instiz.www.instiz', 'com.dcinside.app', 'com.dcinside.app.android', 'com.friendscube.somoim', 'kr.comento.core', 'com.kakao.story', 'me.zepeto.main', 'works.jubilee.timetree', 'kr.co.lawcompany.lawtalk', 'com.pj.banbanvote', 'com.vinulabs.campuspick', 'live.arca.android', 'live.arca.android.playstore', 'com.pulapp.buboo', 'com.ygosu.version2', 'com.postype.play', 'com.yonple.yonple', 'com.everytime.v2', 'bang3.ewhaian.webapp', 'com.cleanbit.joulebug', 'kr.co.rememberapp', 'com.bbasak.bbasakmainapp', 'com.jaivestudio1', 'com.ku.kung', 'com.linker.edulinker', 'com.skt.treal.jumpvrm', 'com.mobions.asked', 'net.daum.android.tistoryapp', 'sky.kr.co.circlina', 'com.dokpa.aos', 'com.remakeing.tree', 'com.atommerce.mindpro', 'atommerce.mindcafe', 'com.unimewo.careclient', 'com.yolo.yeogoo', 'com.toluna.webservice', 'kr.orbi.android', 'com.newreop.newreopapp', 'com.tendom.addcampus', 'kr.univ100.app', 'com.taling', 'kr.nacommunity.anyman', 'kr.villagebaby.billyapplication', 'com.samsungcard.baby', 'com.ajdll5.ivancity', 'com.ppomppu.android', 'kr.co.invenapp', 'com.ulink.ulink', 'com.kunize.uswtimetable']  # Apps used for common interests or objectives
others_custom = ['us.zoom.videomeetings', 'com.cisco.webex.meetings', 'com.gworks.oneapp.naverworks', 'com.kakaoenterprise.kakaowork', 'com.slack', 'com.google.android.gm', 'com.google.android.apps.meetings', 'com.google.android.apps.tachyon', 'com.nhn.android.mail', 'com.samsung.android.email.provider', 'com.skype.raider', 'com.crinity.mail.mobile.hybrid.android', 'mobile.mail.korea.kr', 'com.microsoft.teams', 'com.sidea.hanchon', 'com.dooray.all', 'com.dbs.mthink.khu', 'com.dbs.mthink.uhs', 'com.dbs.mthink.cuk', 'com.classumreactnativeapp.android', 'com.ewha.coursemos.android', 'com.xinics.classmix', 'net.megastudy.qube', 'com.duzon.bizbox.next.phone', 'ru.mail.cloud', 'com.google.android.talk']  # Apps used for work and other purposes
dating_custom = ['com.charmy.cupist', 'com.datingboss.percent', 'net.nrise.wippy', 'kr.co.attisoft.soyou', 'com.endressdreamky.loveting', 'com.waltzdate.go', 'kr.co.teamblind.bleet', 'com.hpcnt.vana', 'app.hybirds.skypeople', 'com.dating.diamatch', 'com.abletree.someday', 'com.tinder', 'com.goldspoon', 'com.bumble.app', 'com.enfpy.app', 'com.badoo.mobile', 'com.hongha.yellowrubberband', 'com.yeonpick.app', 'com.tenfingers.seouldatepop', 'com.ivancity.ivancityapp', 'com.blued.international', 'mobi.jackd.android', 'jstudio.cting', 'com.mobile.sptalks']  # Dating apps

def replace_genre_to_custom():
    """
    Reads a CSV file of apps, updates the 'genre' column based on custom categories, and saves the updated DataFrame.
    """
    df = pd.read_csv('/media/gangnampsy/Elements/dp_230719/entire_apps.csv')
    df.loc[df['package_name'].isin(chat_custom), 'genre'] = 'chat_custom'
    df.loc[df['package_name'].isin(sns_custom), 'genre'] = 'sns_custom'
    df.loc[df['package_name'].isin(community_custom), 'genre'] = 'community_custom'
    df.loc[df['package_name'].isin(others_custom), 'genre'] = 'others_custom'
    df.loc[df['package_name'].isin(dating_custom), 'genre'] = 'dating_custom'
    df.to_csv('/media/gangnampsy/Elements/dp_230719/entire_apps_customized.csv', index=False)

def preprocessing(dataframe, data_type):
    '''
    Preprocesses the dataframe based on the specified data type. 
    Removes out-of-date records, handles specific value conditions, and removes duplicates.

    Args:
    dataframe: The DataFrame to preprocess.
    data_type: Type of data ('location', 'light', 'call', 'application', 'screen').
    '''

    # Converting 'datetime' to the correct format efficiently
    try:
        dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])
    except Exception as e:
        print(f"Error in datetime conversion: {e}")
        return dataframe

    # Removing records with 'datetime' year before 2020
    dataframe = dataframe[dataframe['datetime'].dt.year >= 2020]

    # Handling different data types
    if data_type == 'location':
        dataframe = dataframe[(dataframe['latitude'] != -999) & (dataframe['longitude'] > -999)]
    
    elif data_type == 'light':
        dataframe = dataframe[dataframe['value'] != -999.0]
        
    elif data_type == 'call':
        # If duplicated call log, keep only the max duration row
        dataframe.sort_values(by='duration', ascending=False, inplace=True)
    
    elif data_type == 'application':
        pass  # No additional filtering needed

    elif data_type == 'screen':
        dataframe['type'] = dataframe['type'].replace({15: 1, 16: 0})

    # Removing duplicates and sorting
    dataframe = dataframe.drop_duplicates('datetime', keep='first')
    dataframe.sort_values(by='datetime', inplace=True)    
    dataframe.reset_index(drop=True, inplace=True)
    
    return dataframe

def convert_time(dataframe):
    '''
    Converts 'datetime' column of a DataFrame to Unix timestamp format (milliseconds since epoch).
    '''

    # Resetting index for consistency
    dataframe = dataframe.reset_index(drop=True)

    # Vectorized operation to convert datetime to Unix timestamp in milliseconds
    try:
        # Assuming 'datetime' column is in string format
        dataframe['timestamp'] = pd.to_datetime(dataframe['datetime']).astype(int) // 10**6
    except Exception as e:
        # Handling cases where 'datetime' is not in the expected format
        print(f"Error in converting datetime: {e}")
        dataframe['timestamp'] = dataframe['datetime'].apply(lambda x: int(pd.Timestamp(x).timestamp() * 1000))

    # Removing the original 'datetime' column
    dataframe.drop(columns=['datetime'], inplace=True)

    return dataframe

def clean_application_log(df):
    '''
    Remove abnormal execution and termination records.
    Deletes rows where:
    - An app is turned on but has no corresponding turn off record, or
    - An app is turned off but has no corresponding turn on record.
    '''

    # Ensuring 'datetime' is in datetime format and sorting
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values('datetime', inplace=True)

    # Removing consecutive 'start' (type 1) or 'end' (type 2) records for the same app
    for record_type in [1, 2]:
        is_consecutive = (df['type'] == record_type) & (df['type'].shift(-1 if record_type == 1 else 1) == record_type) & (df['app'] == df['app'].shift(-1 if record_type == 1 else 1))
        df = df[~is_consecutive]
        df.reset_index(drop=True, inplace=True)

    df.sort_values(['app', 'datetime'], inplace=True)
    df['group'] = ((df['type'] == 1) & (df['type'].shift() == 2)).cumsum()
    df.sort_values('datetime', inplace=True)

    # Removing groups with more than two types
    group_counts = df['group'].value_counts()
    invalid_groups = group_counts[group_counts > 2].index
    df = df[~df['group'].isin(invalid_groups)]
    df.reset_index(drop=True, inplace=True)

    return df

def add_usage_duration(dataframe):
    '''
    Adds usage duration to each app usage record.
    '''
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])
    
    # Initialize a dictionary to track the start time of app usage
    type1_dict = {}
    
    # Add columns for start_timestamp, end_timestamp, and duration with default NaN
    dataframe['start_timestamp'] = pd.NaT
    dataframe['end_timestamp'] = pd.NaT
    dataframe['duration'] = pd.NaT
    
   # ipdb.set_trace()

    for index, row in dataframe.iterrows():
        app = row['app']
        type_val = row['type']
        datetime_val = row['datetime']
        
        if type_val == 1:
            # For an 'on' event, update the start datetime for the app
            type1_dict[app] = datetime_val
            dataframe.at[index, 'start_timestamp'] = datetime_val
            
        elif type_val == 2 and app in type1_dict:
            # For an 'off' event, calculate the duration if there is a corresponding 'on' event
            start_datetime = type1_dict.pop(app, pd.NaT)
            if not pd.isna(start_datetime):
                duration = datetime_val - start_datetime
                dataframe.at[index, 'start_timestamp'] = start_datetime
                dataframe.at[index, 'end_timestamp'] = datetime_val
                dataframe.at[index, 'duration'] = duration
    
    # Remove 'on' events and unnecessary columns
    final_dataframe = dataframe.dropna(subset=['duration']).drop(['type'], axis=1).reset_index(drop=True)
    
    return final_dataframe

def assign_group(pid):
    #ipdb.set_trace()
    if pid in hc_mapping.keys():
        return 'hc'
    elif pid in anx_mapping.keys():
        return 'anx'
    else:
        return 'unknown'


def detect_separator(filepath):
    # Expand the '~' to the user's home directory and ensure the filepath is absolute
    absolute_filepath = os.path.abspath(os.path.expanduser(filepath))
    
    # Check if the file exists
    if not os.path.exists(absolute_filepath):
        raise FileNotFoundError(f"No such file or directory: '{absolute_filepath}'")
    
    # Open the file in text mode and read the first line
    with open(absolute_filepath, 'r') as file:
        first_line = file.readline()
    
    # Check if ';' is more common than ',' in the first line
    if first_line.count(';') > first_line.count(','):
        return ';'
    else:
        return ','

def filter_exceeding_hours(df, n):
    filtered_df = df[df['duration'] < pd.Timedelta(hours = n)]
    
    return filtered_df

def create_applications(df, pid, nickname):
    '''
    Prepares application usage data.
    Sorts, preprocesses, cleans, and structures the dataframe for a specific participant identified by pid and nickname.
    '''

    # Checking if DataFrame is not empty
    if not df.empty:
        print(f"Processing applications for PID: {pid}, Nickname: {nickname}")

        # Sorting by datetime and resetting index
        df = df.sort_values('datetime', ascending=True)            
        df.reset_index(drop=True, inplace=True)            

        # Preprocessing: Removing duplicates and wrong dates
        df = preprocessing(df, 'application')

        # Cleaning and adding usage duration
        df = clean_application_log(df)
        df = add_usage_duration(df)
        
        # Keep only duration less than 6 hours
        df = filter_exceeding_hours(df, 6) 
        
        # Converting time to the required format
        df = convert_time(df)

        # Inserting device_id and _id columns and converting _id to integer
        df.insert(1, 'device_id', pid)
        df.insert(0, '_id', pid)
        df['_id'] = df['_id'].astype(int)

        # Renaming columns and initializing additional columns
        df = df.rename(columns={'app': 'package_name'})
        df['application_name'] = ''
        df['is_system_app'] = '0'
        df['label'] = nickname

        # Selecting and ordering columns for the final dataframe
        df = df[['_id', 'timestamp', 'device_id', 'package_name', 'application_name', 'is_system_app', 'duration', 'end_timestamp', 'label']]
        
        if nickname in anx_dict:
            df['group'] = 'anx'
            df['pid'] = anx_dict[nickname]
        elif nickname in hc_dict:
            df['group'] = 'hc'
            df['pid'] = hc_dict[nickname]
        
    else:
        # Handling the case where the dataframe is empty
        print(f"****No application log in {nickname} ({pid})****")

    return df

def create_application_stream(participant_dict, group, version):
    '''
    Processes application log data for each participant in the participant_dict and compiles it into a single DataFrame.
    Exports the compiled data to a CSV file.

    Args:
    participant_dict: Dictionary with participant IDs as keys and nicknames as values.
    group: The group identifier for the output file naming.
    '''
    
    print(f"Processing group {group} for ver {version}")
    

    final_df = pd.DataFrame()

    for nickname, participant_id in tqdm(participant_dict.items()):
        #ipdb.set_trace()
        filepath = f'/media/gangnampsy/Elements/dp_{ver}/DP/{nickname}/ApplicationLog.csv'
        if os.path.exists(filepath):

            separator = detect_separator(filepath)
            df = pd.read_csv(filepath, sep=separator)

            # Processing application data for each participant
            df = create_applications(df, participant_id, nickname)

            # Concatenating the processed data
            final_df = pd.concat([df, final_df], ignore_index=True)

        else:
            # If the file does not exist, skip to the next iteration
            print(f"File {filepath} (pid: {participant_id}) not found, skipping.")

    # Removing unnecessary columns
    columns_to_remove = ['datetime', 'app', 'type']
    final_df.drop(columns=[col for col in columns_to_remove if col in final_df], inplace=True)

    # Converting data types for consistency
    final_df['device_id'] = final_df['device_id'].astype(int)
    final_df['_id'] = final_df['_id'].astype(int)
    
    #final_df.insert(1, 'group', group)

    # Exporting the compiled DataFrame to a CSV file
    output_filepath = f'~/Desktop/DP/re_analysis/raw_stream/applications_{group}_{ver}.csv'
    final_df.to_csv(output_filepath, index=False)
    
    print(f"Number of unique IDs in {group}: ", final_df['_id'].nunique())
    
    return final_df



# main execution
group = 'both' #  'hc', 'anx', or 'both'
ver = '240205' #  '230719' or '240205' 

if group == 'hc':
    p_dict = hc_dict
elif group == 'anx':
    p_dict = anx_dict
elif group == 'both':
    p_dict = total_dict

create_application_stream(p_dict, group, ver)
