import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

plt.rcParams['figure.figsize'] = (10.0, 5.0)
plt.figure(figsize=(20, 6))

FILE_CARD_CLEAN = 'data/data_card_clean.csv'
FILE_TRAIN = 'data/train_timetable_data.csv'
FILE_RESULT = 'data/result.csv'
FILE_RESULT_31 = 'data/result_10_31.csv'
FILE_RESULT_CLEAN = 'data/result_clean.csv'
FILE_RESULT_DIRTY = 'data/result_dirty.csv'
FILE_RESULT_FINAL = 'data/result_final.csv'
FILE_RESULT_FINAL_TIMESTAMP = 'data/result_final_timestamp.csv'
FILE = 'data/result_final_timestamp_10_31.csv'
FILE_TRAIN = 'data/result_10_31_371033.csv'
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
    df = pd.read_csv('data/result_10_31_371033.csv')
    if part:
        df = df.loc[:100, ]
    return df


def plt_joint_graph(df, x_name, y_name, x_label, y_label, title):
    g = sns.JointGrid(data=df, x=x_name, y=y_name)
    g.plot_joint(sns.histplot,discrete=(True, True), cbar=True)
    g.plot_marginals(sns.histplot, kde=True)
    ax = plt.gca()
    ax.set_title(title)
    g.set_axis_labels(xlabel=x_label, ylabel=y_label)
    plt.tight_layout()
    plt.show()


state_flow = pd.DataFrame(columns=['station', 'timestamp'])


def write_state(card):
    sta_time = card['ENTRY_TIME_TIME_STAMP']
    des_time = card['DEAL_TIME_TIME_STAMP']
    travel = card['DESTINATION_STATION'] - card['ORIGIN_STATION']
    for i in range(card['ORIGIN_STATION'], card['DESTINATION_STATION']):
        tim = sta_time + (des_time - sta_time) * ((i - card['ORIGIN_STATION']) / travel)

        state_flow.loc[(len(state_flow))] = [i, tim]


def check_2D_FLOW_WITH_STA(df):
    df['ENTRY_TIME_TIME_STAMP'] = df.apply(lambda x: pd.Timestamp(pd.to_datetime(x['ENTRY_TIME'])), axis=1)
    df['DEAL_TIME_TIME_STAMP'] = df.apply(lambda x: pd.Timestamp(pd.to_datetime(x['DEAL_TIME'])), axis=1)
    df.apply(write_state, axis=1)
    plt_joint_graph(state_flow, "station", "timestamp", "STATION", "TIME", "FLOW")




def plt_entry_station(df):
    df['ENTRY_TIME_TIME_STAMP'] = df.apply(lambda x: pd.Timestamp(pd.to_datetime(x['ENTRY_TIME'])), axis=1)
    plt_joint_graph(df, "ORIGIN_STATION", "ENTRY_TIME_TIME_STAMP", "STATION", "TIME", "Entry Station")


def plt_out_station(df):
    df['DEAL_TIME_TIME_STAMP'] = df.apply(lambda x: pd.Timestamp(pd.to_datetime(x['DEAL_TIME'])), axis=1)
    plt_joint_graph(df, "DESTINATION_STATION", "DEAL_TIME_TIME_STAMP", "STATION", "TIME", "Out Station")


def read_train():
    train = pd.read_csv('data/371033.csv', index_col='STATION')
    train['DEPATURETIME'] = train.apply(lambda x: pd.Timestamp(pd.to_datetime(x['DEPATURETIME'])).timestamp(), axis=1)
    train['ARRIVETIME'] = train.apply(lambda x: pd.Timestamp(pd.to_datetime(x['ARRIVETIME'])).timestamp(), axis=1)
    return train


def clean_371033():
    train = read_train()
    # train.to_csv('data/371033.csv')
    df = read_result_final(False)
    df.sort_values(by=['ENTRY_TIME'], ignore_index=True, inplace=True)

    def hahaha(x):
        time1 = pd.Timestamp.fromtimestamp(train.loc[x['ORIGIN_STATION']]['DEPATURETIME'])
        time2 = pd.Timestamp.fromtimestamp(x['ENTRY_TIME_TIME_STAMP'])
        return 60 * (time1.hour - time2.hour) + time1.minute - time2.minute + 1 / 60 * (time1.second - time2.second)

    df['wait_time'] = df.apply(hahaha, axis=1)
    df.sort_values(by=['wait_time'])
    df.to_csv('clean.csv')


state_flow_2 = pd.DataFrame(columns=['station'])


def write_num_new(card):
    for i in range(card['ORIGIN_STATION'], card['DESTINATION_STATION']):
        state_flow_2.loc[(len(state_flow_2))] = [i]


def write_371033_people():
    df = read_result_final(False)
    df.apply(write_num_new, axis=1)

    print(state_flow_2)
    sns.histplot(state_flow_2, bins=11, kde=True)
    plt.xlabel('Station')
    plt.ylabel('Number of people')
    plt.title('Number of people in 371033')
    plt.show()


train = read_train()

state_flow_3 = pd.DataFrame(columns=['station', 'time'], )


def write_time(x):
    time_out = pd.Timestamp.fromtimestamp(train.loc[x['ORIGIN_STATION']]['DEPATURETIME'], 'UTC').replace(year=2022,
                                                                                                         month=10,
                                                                                                         day=31)
    time_in = pd.Timestamp.fromtimestamp(x['ENTRY_TIME_TIME_STAMP'], 'UTC')
    time_out_stamp = time_out.timestamp()
    time_in_stamp = time_in.timestamp()

    for i in range(int(time_in_stamp), int(time_out_stamp), 30):
        time = pd.Timestamp.fromtimestamp(i, 'UTC')
        state_flow_3.loc[(len(state_flow_3))] = [x['ORIGIN_STATION'], time]


def write_waiting_because_of_371033():
    df = read_result_final(False)
    df = df[df['NUM'] <= 2]
    df.apply(write_time, axis=1)

    plt_joint_graph(state_flow_3, 'station', 'time', 'STATION', 'TIME', 'WAITING PEOPLE')


# write_waiting_because_of_371033()
def plt_hot():
    df = read_result_final(False)
    df_slice = df[["ORIGIN_STATION", 'DESTINATION_STATION', 'GRANT_CARD_CODE']]
    print(df_slice)
    df_groupby = df_slice.groupby(["ORIGIN_STATION", 'DESTINATION_STATION'], as_index=False).count()
    print(df_groupby)
    # f_groupby['GRANT_CARD_CODE']=np.log10(df_groupby['GRANT_CARD_CODE'])
    print(df_groupby)
    print("---------------------------------")
    df_pivot = df_groupby.pivot(index='DESTINATION_STATION', columns="ORIGIN_STATION", values="GRANT_CARD_CODE")
    print(df_pivot.columns)
    sns.set_context({"figure.figsize": (8, 8)})
    sns.heatmap(df_pivot, square=True)
    plt.show()


def alway_train():
    df = read_result_final(True)
    df.sort_values(by=['GRANT_CARD_CODE'], )
    print(df)
    df_slice = df[['GRANT_CARD_CODE', 'ORIGIN_STATION', 'DESTINATION_STATION']]
    print(df_slice)


def demp():
    train_travel = np.zeros((12, 12))

    def ao(x):
        return x["TRAVEL_MINUTE"] - train_travel[x['ORIGIN_STATION'] - 1][x['DESTINATION_STATION'] - 1]

    df = pd.read_csv('data/result_dirty.csv')
    train = read_train()
    train_travel = np.zeros((12, 12))

    for i in range(0, 12):
        for j in range(i + 1, 12):
            train_travel[i][j] = (train.loc[j + 1]['ARRIVETIME'] - train.loc[i + 1]['DEPATURETIME']) / 60
    df['MINUS'] = df.apply(ao, axis=1)
    print(df['MINUS'])
    sns.histplot(df['MINUS'], bins=100, kde=True)
    plt.xlabel('time')
    plt.ylabel('Frequency')
    plt.title('time different in real travel time and predict travel time')
    plt.show()

def station_wait():
    df = pd.read_csv('data/result_10_31_371033.csv')
    plt_joint_graph(df,x_name='ORIGIN_STATION',y_name='NUM',x_label='origin station',y_label= 'wait train',title='haha')

station_wait()