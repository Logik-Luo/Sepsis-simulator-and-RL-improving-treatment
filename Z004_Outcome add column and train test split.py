import random
import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/A002_MyData.csv')
df = df.reset_index(drop=True)
df.rename(columns={'Unnamed: 0': 'row_id'}, inplace = True)

df['row_id_next'] = df['row_id'] + 1

ids = df['Admn001_ID'].unique()
# print(df['Admn001_ID'].nunique())
train_ids = random.sample(list(ids), 16677)        # frac = 0.8 equals n=16676.8
test_ids = np.setdiff1d(ids, train_ids)
df['iv_input'] = df['iv_input']+1
df['vaso_input'] = df['vaso_input']+1

df['discrete_action'] = 0

df.loc[((df['iv_input'] == 0) & (df['vaso_input'] == 0)), 'discrete_action'] = 1        # no drug
df.loc[((df['iv_input'] != 0) & (df['vaso_input'] == 0)), 'discrete_action'] = 2        # IV only
df.loc[((df['iv_input'] == 0) & (df['vaso_input'] != 0)), 'discrete_action'] = 3        # Vaso only
df.loc[((df['iv_input'] != 0) & (df['vaso_input'] != 0)), 'discrete_action'] = 4

df['reward'] = 0
# df.loc[(df['OutC002_90d mortality'] == 1), 'reward'] = -15
# df.loc[(df['OutC002_90d mortality'] == 0), 'reward'] = 15
# Redefine reward
last_steps = df.groupby('Admn001_ID')['bloc'].idxmax()      # bloc最大值获得
df.loc[last_steps, 'reward'] = df.loc[last_steps, 'OutC002_90d mortality'] + 1
df['reward'] = df['reward'].round()
print(df['reward'].value_counts().sort_values(ascending=False))

row_id = df.pop('row_id')
df.insert(loc=df.shape[1], column='row_id', value=row_id, allow_duplicates=False)
row_idnext = df.pop('row_id_next')
df.insert(loc=df.shape[1], column='row_id_next', value=row_idnext, allow_duplicates=False)
Mortality = df.pop('OutC002_90d mortality')
df.insert(loc=df.shape[1], column='OutC002_90d mortality', value=Mortality, allow_duplicates=False)
timeday = df.pop('timeday')
df.insert(loc=df.shape[1], column='timeday', value=timeday, allow_duplicates=False)
df = df.drop(['Flud002_Input4H', 'Flud003_MaxVaso'], axis=1)

train_id_df = df[df.Admn001_ID.isin(train_ids)]
test_id_df = df[df.Admn001_ID.isin(test_ids)]

train_id_df.to_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/train_state_action_reward_df.csv')
test_id_df.to_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/test_state_action_reward_df.csv')
