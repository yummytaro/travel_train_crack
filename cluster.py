import pandas as pd

FILE = 'data/result_final_timestamp.csv'


def read_card(pre=False):
    df = pd.read_csv(FILE)
    if pre:
        df = df.loc[:1000, ]
    return df


df = read_card(False)
df_fre_cus = pd.DataFrame(columns=['CARDID', 'ORI_STA', 'DES_STA', 'COUNTS'])
df_fre = pd.DataFrame(columns=list(df).append('NUM'))
df_rare = pd.DataFrame(columns=list(df).append('NUM'))

df_group = df.groupby(['GRANT_CARD_CODE', 'ORIGIN_STATION', 'DESTINATION_STATION'])
for (card_id, ori_sta, des_sta), group in df_group:
    group['NUM'] = len(group.index)
    if (len(group.index) >= 9):
        df_fre = pd.concat([df_fre, group])
        df_fre_cus.loc[(len(df_fre_cus))] = [card_id, ori_sta, des_sta, len(group.index)]
    else:
        df_rare = pd.concat([df_rare, group])

print(df_fre)
print(df_fre_cus)
print(df_rare)

df_fre.to_csv('data/q3_fre_data.csv', index=False)
df_rare.to_csv('data/q3_rare_data.csv', index=False)
df_fre_cus.to_csv('data/q3_fre_cus.csv', index=False)
