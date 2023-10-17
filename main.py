import pandas as pd
import numpy as np
import time

FILE_CARD_CLEAN = 'data/data_card_clean.csv'
FILE_TRAIN = 'data/train_timetable_data.csv'
FILE_SAVE = 'data/result.csv'
sta_list = {}
def read_train():
    df = pd.read_csv(FILE_TRAIN)
    stations = df['STATION'].unique()


    for sta in stations:
        _df = df[df['STATION'] == sta].sort_values(by="ARRIVETIME")
        sta_list[sta] = _df


def read_card():
    df_card = pd.read_csv(FILE_CARD_CLEAN)
    return df_card

def train_statis_time(sta,time_start,time_end,up = True):
    df = sta_list[sta]
    df['DEPATURETIME'] = pd.to_datetime(df['DEPATURETIME'], format="%H:%M:%S").dt.time
    df['ARRIVETIME'] = pd.to_datetime(df['ARRIVETIME'], format="%H:%M:%S").dt.time
    df_sta = df[(df['DEPATURETIME']>= time_start) & (df['ARRIVETIME']<= time_end)]
    list = df_sta['TRAINID'].values
    return list


def check_enter_time(card):
    ori_sta = card['ORIGIN_STATION']
    des_sta = card['DESTINATION_STATION']
    ori_tim = pd.to_datetime(card['ENTRY_TIME'], format="%Y-%m-%d %H:%M:%S").time()
    des_tim =  pd.to_datetime(card['DEAL_TIME'], format="%Y-%m-%d %H:%M:%S").time()
    # ori_tim = card['ENTRY_TIME']
    # des_tim = card['DEAL_TIME']
    ori_list = train_statis_time(ori_sta,ori_tim,des_tim,up=True)
    des_list = train_statis_time(des_sta,ori_tim,des_tim,up=False)
    list = np.intersect1d(des_list,ori_list)
    if card.name%100 == 0:
        print(str(card.name)+" finished")
    return list



read_train()
card = read_card()
print(card.size)
#card = card.loc[1:10000,]
card['LIST'] = card.apply(check_enter_time,axis=1)
card.to_csv(FILE_SAVE,index =False)



