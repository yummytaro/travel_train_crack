import pandas as pd
import numpy as np
import datetime as dt

FILE_CARD_CLEAN = 'data/data_card_clean.csv'
FILE_TRAIN = 'data/train_timetable_data.csv'
FILE_SAVE = 'data/result.csv'
sta_list = {}


def read_train():
    df = pd.read_csv(FILE_TRAIN)
    stations = df['STATION'].unique()

    for sta in stations:
        _df = df[df['STATION'] == sta].sort_values(by="ARRIVETIME")
        _df['ARRIVETIME'] = pd.to_datetime(df['ARRIVETIME'], format="%H:%M:%S").dt.time
        _df['DEPATURETIME'] = pd.to_datetime(df['DEPATURETIME'], format="%H:%M:%S").dt.time
        sta_list[sta] = _df


def read_card():
    df_card = pd.read_csv(FILE_CARD_CLEAN)
    df_card['ENTRY_TIME'] = pd.to_datetime(df_card['ENTRY_TIME'], format="%Y/%m/%d %H:%M")
    df_card['DEAL_TIME'] = pd.to_datetime(df_card['DEAL_TIME'], format="%Y/%m/%d %H:%M")
    return df_card


def train_statis_time(sta, time_start, time_end):
    df = sta_list[sta]
    df_sta = df[(df['DEPATURETIME'] >= time_start) & (df['ARRIVETIME'] <= time_end)]
    list = df_sta['TRAINID'].values
    return list


def check_enter_time(card):
    ori_sta = card['ORIGIN_STATION']
    des_sta = card['DESTINATION_STATION']
    ori_tim = card['ENTRY_TIME']
    des_tim = card['DEAL_TIME']
    ori_tim = (ori_tim + dt.timedelta(seconds=-30)).time()
    des_tim = (des_tim + dt.timedelta(seconds=30)).time()
    ori_list = train_statis_time(ori_sta, ori_tim, des_tim)
    des_list = train_statis_time(des_sta, ori_tim, des_tim)
    list_train = np.intersect1d(des_list, ori_list)
    if card.name % 100 == 0:
        print(str(card.name) + " finished")
    return list_train


read_train()
card = read_card()
card = card.loc[1:100, ]
card['LIST'] = card.apply(check_enter_time, axis=1)
card.to_csv(FILE_SAVE, index=False)
