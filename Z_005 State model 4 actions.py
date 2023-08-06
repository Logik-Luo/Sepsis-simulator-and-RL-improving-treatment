import random
import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/A002_MyData.csv')
df = df.reset_index(drop=True)
df.rename(columns={'Unnamed: 0': 'row_id'}, inplace = True)

df['row_id_next'] = df['row_id'] + 1

# df['iv_input'] = df['iv_input']+16676
# df['vaso_input'] = df['vaso_input']+1
df['discrete_action'] = 0

df.loc[((df['iv_input'] == 0) & (df['vaso_input'] == 0)), 'discrete_action'] = 1        # no drug
df.loc[((df['iv_input'] != 0) & (df['vaso_input'] == 0)), 'discrete_action'] = 2        # IV only
df.loc[((df['iv_input'] == 0) & (df['vaso_input'] != 0)), 'discrete_action'] = 3        # Vaso only
df.loc[((df['iv_input'] != 0) & (df['vaso_input'] != 0)), 'discrete_action'] = 4        # both
row_id = df.pop('row_id')
df.insert(loc=df.shape[1], column='row_id', value=row_id, allow_duplicates=False)
row_idnext = df.pop('row_id_next')
df.insert(loc=df.shape[1], column='row_id_next', value=row_idnext, allow_duplicates=False)
Mortality = df.pop('OutC002_90d mortality')
df.insert(loc=df.shape[1], column='OutC002_90d mortality', value=Mortality, allow_duplicates=False)
timeday = df.pop('timeday')
df.insert(loc=df.shape[1], column='timeday', value=timeday, allow_duplicates=False)
df = df.drop(['Flud002_Input4H', 'Flud003_MaxVaso'], axis=1)

df_1 = df[df['discrete_action'].isin([1])]
df_2 = df[df['discrete_action'].isin([2])]
df_3 = df[df['discrete_action'].isin([3])]
df_4 = df[df['discrete_action'].isin([4])]
# print(df_1['Admn001_ID'].nunique())     # 18890 * 0.8 = 15112
# print(df_2['Admn001_ID'].nunique())     # 18024 * 0.8 = 14419.2
# print(df_3['Admn001_ID'].nunique())     # 1782 * 0.8 = 1425.6
# print(df_4['Admn001_ID'].nunique())     # 5902 * 0.8 = 4721.6

ids1 = df_1['Admn001_ID'].unique()
ids2 = df_2['Admn001_ID'].unique()
ids3 = df_3['Admn001_ID'].unique()
ids4 = df_4['Admn001_ID'].unique()

train_ids1 = random.sample(list(ids1), 15112)
test_ids1 = np.setdiff1d(ids1, train_ids1)

train_ids2 = random.sample(list(ids2), 14419)
test_ids2 = np.setdiff1d(ids2, train_ids2)

train_ids3 = random.sample(list(ids3), 1426)
test_ids3 = np.setdiff1d(ids3, train_ids3)

train_ids4 = random.sample(list(ids4), 4722)
test_ids4 = np.setdiff1d(ids4, train_ids4)

train_id_df1 = df_1[df_1.Admn001_ID.isin(train_ids1)]
test_id_df1 = df_1[df_1.Admn001_ID.isin(test_ids1)]

train_id_df2 = df_2[df_2.Admn001_ID.isin(train_ids2)]
test_id_df2 = df_2[df_2.Admn001_ID.isin(test_ids2)]

train_id_df3 = df_3[df_3.Admn001_ID.isin(train_ids3)]
test_id_df3 = df_3[df_3.Admn001_ID.isin(test_ids3)]

train_id_df4 = df_4[df_4.Admn001_ID.isin(train_ids4)]
test_id_df4 = df_4[df_4.Admn001_ID.isin(test_ids4)]

train_id_df1.to_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/train_state_action_reward_df1.csv')
test_id_df1.to_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/test_state_action_reward_df1.csv')

train_id_df2.to_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/train_state_action_reward_df2.csv')
test_id_df2.to_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/test_state_action_reward_df2.csv')

train_id_df3.to_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/train_state_action_reward_df3.csv')
test_id_df3.to_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/test_state_action_reward_df3.csv')

train_id_df4.to_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/train_state_action_reward_df4.csv')
test_id_df4.to_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/test_state_action_reward_df4.csv')