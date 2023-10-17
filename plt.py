import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_CARD_CLEAN = 'data/data_card_clean.csv'
FILE_TRAIN = 'data/train_timetable_data.csv'


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
        time.apend(tim/pd.Timedelta(minutes=1))
    print(time)
    sns.histplot(time, bins=10, kde=True)
    plt.xlabel('Train_Travel_Time (mintue)')
    plt.ylabel('Frequency')
    plt.title('Train Travel Time Frequency')
    plt.show()


