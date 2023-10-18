import pandas as pd

FILE = 'data/smart_card_data_merge.csv'
FILE_DIRTY_ARRIVE = 'data/dirty_arrive.csv'
FILE_CARD_CLEAN = 'data/data_card_clean.csv'


def read_dataframe(src):
    df = pd.read_csv(src)
    df['ENTRY_TIME'] = pd.to_datetime(df['ENTRY_TIME'], format="%Y/%m/%d %H:%M")
    df['DEAL_TIME'] = pd.to_datetime(df['DEAL_TIME'], format="%Y/%m/%d %H:%M")
    df["TRAVEL"] = df['DEAL_TIME'] - df['ENTRY_TIME']
    df['TRAVEL_MINUTE'] = df['TRAVEL'] / pd.Timedelta(minutes=1)
    df['DIRTY'] = 'CLEAN'
    df['DIRTY'] = df.apply(lambda x: x['DIRTY'] if x['ENTRY_TIME'] < x['DEAL_TIME'] else 'WRONG ENTER TIME', axis=1)
    df['DIRTY'] = df.apply(lambda x: x['DIRTY'] if x['ORIGIN_STATION'] < x['DESTINATION_STATION'] else 'WRONG STATION',
                           axis=1)
    df['DIRTY'] = df.apply(lambda x: x['DIRTY'] if x['TRAVEL_MINUTE'] < 100 else 'TO LONG TRAVEL TIME',
                           axis=1)

    df_dirty = df[df['DIRTY'] != 'CLEAN']
    df_clean = df[df['DIRTY'] == 'CLEAN']

    return df_clean, df_dirty


df, df_dirty = read_dataframe(FILE)

df_dirty.to_csv(FILE_DIRTY_ARRIVE, index=False)
df.to_csv(FILE_CARD_CLEAN, index=False)
