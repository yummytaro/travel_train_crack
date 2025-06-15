import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

FILE_RESULT = 'data/result_final_timestamp_10_31.csv'
FILE_TRAIN = 'data/train_timetable_data.csv'

sta_list = {}


def plt_joint_graph(df, x_name, y_name, x_label, y_label, title):
    g = sns.JointGrid(data=df, x=x_name, y=y_name)
    g.plot_joint(sns.scatterplot, s=100, alpha=.002)
    g.plot_marginals(sns.histplot, kde=True)
    ax = plt.gca()
    ax.set_title(title)
    g.set_axis_labels(xlabel=x_label, ylabel=y_label)
    plt.tight_layout()
    plt.show()


def read_train_new():
    df = pd.read_csv(FILE_TRAIN)
    stations = df['STATION'].unique()

    for sta in stations:
        _df = df[df['STATION'] == sta].sort_values(by="ARRIVETIME")
        _df['DEPATURETIME'] = _df.apply(lambda x: pd.Timestamp(pd.to_datetime(x['DEPATURETIME'])).timestamp(), axis=1)
        _df['ARRIVETIME'] = _df.apply(lambda x: pd.Timestamp(pd.to_datetime(x['ARRIVETIME'])).timestamp(), axis=1)
        sta_list[sta] = _df


def read_CARD():
    train = pd.read_csv(FILE_RESULT)
    # train = train.loc[:1000
    # ,]
    return train


state_flow = pd.DataFrame(columns=['station', 'timestamp'])


def write_time(x):
    train = sta_list[x['ORIGIN_STATION']]
    timestamp = int(train.loc[train['TRAINID'] == x['TRAIN']]['ARRIVETIME'])
    time_out = pd.Timestamp.fromtimestamp(timestamp, 'UTC').replace(year=2022, month=10, day=31)
    time_in = pd.Timestamp.fromtimestamp(x['ENTRY_TIME_TIME_STAMP'], 'UTC')
    time_out_stamp = time_out.timestamp()
    time_in_stamp = time_in.timestamp()

    for i in range(int(time_in_stamp), int(time_out_stamp), 30):
        time = pd.Timestamp.fromtimestamp(i, 'UTC')
        state_flow.loc[(len(state_flow))] = [x['ORIGIN_STATION'], time]


def write_waiting_because_of_371033(df):
    df.apply(write_time, axis=1)
    plt_joint_graph(state_flow, 'station', 'timestamp', 'STATION', 'TIME', 'WAITING PEOPLE')


read_train_new()
write_waiting_because_of_371033(read_CARD())
