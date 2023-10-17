import pandas as pd
import json

FILE = 'data/result.csv'

df = pd.read_csv(FILE)
for i in range(0,10):
    series = df['LIST'][i]
    series = series.replace('\'','\"')
    l = json.loads(series)
    train_list = l['list']
    print(train_list)