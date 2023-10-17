import pandas as pd

FILE_TRAIN = 'data/train_timetable_data.csv'
FILE = 'data/train_{}.csv'
df = pd.read_csv(FILE_TRAIN)
stations = df['STATION'].unique()
sta_list = {}

for sta in stations:
    _df = df[df['STATION'] == sta].sort_values(by="ARRIVETIME")
    sta_list[sta] = _df
    _df.to_csv(FILE.format(str(sta)),index=False)


