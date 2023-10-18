import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_CARD_CLEAN = 'data/data_card_clean.csv'
FILE_TRAIN = 'data/train_timetable_data.csv'
FILE_RESULT = 'data/result.csv'
FILE_RESULT_31 = 'data/result_10_31.csv'
FILE_RESULT_CLEAN = 'data/result_clean.csv'
FILE_RESULT_DIRTY = 'data/result_dirty.csv'
FILE_RESULT_FINAL = 'data/result_final.csv'
FILE_RESULT_FINAL_TIMESTAMP = 'data/result_final_timestamp.csv'
FILE = 'data/result_final_timestamp_10_31.csv'
import json


def plt_travel_time_frequency():
    df = pd.read_csv(FILE_CARD_CLEAN)
    data = np.log10(df['TRAVEL_MINUTE'])
    sns.histplot(data, bins=100, kde=True)
    plt.xlabel('Travel_time (mintue,log10)')
    plt.ylabel('Frequency')
    plt.title('Travel Time Frequency')
    plt.show()


def plt_train_travel_time():
    time = []
    df = pd.read_csv(FILE_TRAIN)
    for name, group in df.groupby("TRAINID"):
        stations = group['STATION'].unique()
        start = stations.min()
        end = stations.max()
        start_sta = group.loc[group['STATION'] == start].reset_index()
        end_sta = group.loc[group['STATION'] == end].reset_index()
        arr_tim = start_sta.loc[0]['ARRIVETIME']
        end_tim = end_sta.loc[0]['DEPATURETIME']
        tim = pd.to_datetime(end_tim, format="%H:%M:%S") - pd.to_datetime(arr_tim, format="%H:%M:%S")
        time.apend(tim / pd.Timedelta(minutes=1))
    print(time)
    sns.histplot(time, bins=10, kde=True)
    plt.xlabel('Train_Travel_Time (mintue)')
    plt.ylabel('Frequency')
    plt.title('Train Travel Time Frequency')
    plt.show()


def _read_json(card):
    series = card['LIST']
    series = series.replace('\'', '\"')
    l = json.loads(series)
    return len(l['list'])


def _read_result():
    df = pd.read_csv(FILE_RESULT)
    df['NUM'] = df.apply(_read_json, axis=1)
    df_part = df[df['NUM'] >= 1]
    df_dirty = df[df['NUM'] == 0]
    print(df_part)
    print(df_dirty)
    df_part.to_csv(FILE_RESULT_CLEAN, index=False)
    df_dirty.to_csv(FILE_RESULT_DIRTY, index=False)


def _read_json_final(card):
    series = card['LIST']
    series = series.replace('\'', '\"')
    l = json.loads(series)
    return l['list'][-1]


def _read_result_final():
    df = pd.read_csv(FILE_RESULT_CLEAN)
    # df = df.loc[:100, ]
    df['TRAIN'] = df.apply(_read_json_final, axis=1)
    df.to_csv(FILE_RESULT_FINAL)
    return df


def read_result_final(part=False):
    df = pd.read_csv(FILE)
    if part:
        df = df.loc[:1000, ]
    return df


flow_with_sta = []


def check_2D_FLOW_WITH_STA(card):
    x = 1


df = read_result_final(False)
df['ENTRY_TIME_TIME_STAMP'] = df.apply(lambda x:pd.Timestamp(x['ENTRY_TIME']).timestamp(),axis=1)
df['DEAL_TIME_TIME_STAMP'] = df.apply(lambda x:pd.Timestamp(x['DEAL_TIME']).timestamp(),axis=1)
df.to_csv(FILE_RESULT_FINAL_TIMESTAMP)
#
# print(df.columns)
# g = sns.JointGrid(data=df, x="ORIGIN_STATION", y="ENTRY_HOUR")
#
# ax = plt.gca()
# # get current xtick labels
# yticks = ax.get_yticks()
# print(yticks)
# # convert all xtick labels to selected format from ms timestamp
# ax.set_yticklabels([pd.to_datetime(tm, unit='ms').strftime('%Y-%M-%D %H:%M:%S') for tm in yticks],
#  rotation=50)
#
# g.plot_joint(sns.scatterplot, s=100, alpha=.0005)
# g.plot_marginals(sns.histplot, kde=True)
# plt.show()
#
#
