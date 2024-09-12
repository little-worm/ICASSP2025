import ibc_api.utils as ibc

# Fetch info on all available files
# Load as a pandas dataframe and save as ibc_data/available_{data_type}.csv 
db = ibc.get_info(data_type="volume_maps")

# Keep statistic maps for sub-08, for task-Discount
#filtered_db = ibc.filter_data(db, subject_list=["08"], task_list=["Discount"])

my_task_list = ["ArchiEmotional","ArchiSocial","ArchiSpatial","ArchiStandard",
                "Attention","Audi","Audio",
                "Bang","ColumbiaCards","Discount","DotPatterns","EmotionalPain","Enumeration","Lec1","Lec2","MCSE","MTTNS",
                "MTTWE","MVEB","MVIS","Moto",
                "PainMovie","Preference","PreferenceFaces","PreferenceFood","PreferenceHouses","PreferencePaintings","RSVPLanguage","SelectiveStopSignal","Self",
                "StopSignal","Stroop","TheoryOfMind","TwoByTwo","VSTM","Visu","WardAndAllport",
                "HcpEmotion","HcpGambling","HcpLanguage","HcpMotor","HcpRelational","HcpSocial","HcpWm",
                "MathLanguage"]
my_subject_list=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
# my_subject_list=['15']
# my_subject_list=['01', '02', '04', '05', '06', '07', '08', '09', '11', '12', '13', '14', '15']

filtered_db = ibc.filter_data(db, subject_list=my_subject_list, task_list=my_task_list)

# Download all statistic maps for sub-08, task-Discount 
# Also creates ibc_data/downloaded_volume_maps.csv 
# which contains local file paths and time of download
downloaded_db = ibc.download_data(filtered_db)