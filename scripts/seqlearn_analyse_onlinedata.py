'''First pass analysing behavioral data'''

#%%
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import src.behavior_preprocess as prep

#%% GET ALL DATA INTO DATAFRAME
PATH_DATA = "data/v3/data"
PATH_DEMOGRAPHIC = "data/prolific/demographics"
PATH_RESULTS = "data/v3/preprocessed"

df_subject = prep.fetch_demographics(PATH_DEMOGRAPHIC, PATH_DATA)

all_data = []
for file in os.listdir(PATH_DATA):
    PATH_TO_FILE = '/'.join([PATH_DATA, file])
    if not PATH_TO_FILE.endswith('.txt'):
        continue
    #retrieve data and put into large pandas dataframe
    data_dicts = prep.data2dict(PATH_TO_FILE)
    all_data.append(prep.dicts2df(data_dicts))

all_data = pd.concat(all_data).reset_index(drop=True)
all_data = all_data.drop(0)
all_data.columns = [str.split('-', 1)[1] for str in all_data.columns]
all_data = all_data.fillna(value=np.nan)

#%% ACCURACY
#how many submissions per
col_group = ['expt_group', 'expt_curriculum', 'expt_block']
col_sel = col_group + ['resp_correct']
all_data.groupby(col_group)['expt_subject'].nunique()

#extract correct/incorrect per participant & block
col_group = ['expt_turker', 'expt_block', 'expt_index']
np_acc = prep.pd2np(all_data, col_group,'resp_correct')
np.nanmean(np_acc, axis=2)

#%% Check for repeat participants
submissions = all_data['expt_subject'].unique()
subjs = all_data['expt_turker'].unique()
counts = Counter(all_data['expt_turker'])

with open('data_log.txt', "r") as file:
    log = file.read().split('\n')
df_log = pd.DataFrame([i.split(' ')[1:] for i in log])

if len(subjs) < len(submissions):
    print("Some subjects participated more than once")
    for sub in subjs:
        if counts[sub] > 329:
            print(sub)
            files = all_data[all_data.expt_turker == sub]['expt_subject'].unique()
            for file in files:
                idx_match = df_log.iloc[:, -1] == file+'.txt'
                time = df_log[idx_match].iloc[:, -2].values
                day = df_log[idx_match].iloc[:, -3].values
                print('{0} completed at {1} on {2}'.format(file, time[0], day[0]))

#%%
##############################################
# First pass, quick stats for each submission, quality check
all_subj = list(set(all_data['expt_subject']))
subj = all_subj[0]
for subj in all_subj:
    print(subj)
    #if int(df_log[df_log.iloc[:, -1] == subj + '.txt'][9].values[0]) < 30:
    #    continue
    prep.output_submission_details(all_data, subj)

#%% Save bonus to file
with open('bonus.csv', 'w') as out:
    datasel = all_data[(all_data.expt_group == 'simple') & (all_data.expt_curriculum == 'interleaved')]
    len(datasel.groupby('expt_subject'))
    for subj, subj_data in datasel.groupby('expt_subject'):
        if int(df_log[df_log.iloc[:, -1] == subj + '.txt'][9].values[0]) < 29:
            continue
        prolific_id = subj_data['expt_turker'].unique()
        bonus_vec = np.array(subj_data['block_bonus'].iloc[0])
        bonus_vec = bonus_vec[bonus_vec != np.array(None)]
        bonus = sum([bonus for bonus in bonus_vec if bonus > 0])

        group = subj_data['expt_group'].unique()[0]
        curr = subj_data['expt_curriculum'].unique()[0]
        subid = subj_data['expt_subject'].unique()[0]
        out.write('{0},{1}\n'.format(prolific_id[0], bonus, group, curr, subid))
        #out.write('{4} {2}-{3} {0},{1}\n'.format(subj, bonus, group, curr, subid))

#%% Preprocessing
# add info time-outs
timeouts = all_data.groupby(['expt_turker'], as_index=False)['resp_correct'].apply(lambda x: pd.Series({'n_timeout':
                                                                               x.isnull().sum()}))
all_data = pd.merge(all_data,timeouts, on='expt_turker')

df_acc = all_data.groupby(['expt_turker', 'expt_block'], as_index=False)['resp_correct'].apply(lambda x: pd.Series({'acc':
                                                                x.sum()/x.count()}))

df_learner = df_acc.groupby(['expt_turker'], as_index=False).apply(lambda x: pd.Series({'learned_1':
                                                x['acc'].values[0] < x['acc'].values[1],
                                                'learned_2': x['acc'].values[2] < x['acc'].values[3],
                                                'learned_3': x['acc'].values[4] < x['acc'].values[5],
                                                'learned_4': x['acc'].values[6] < x['acc'].values[7]}))
all_data = pd.merge(all_data,df_learner, on='expt_turker')

#%% Save data

with open(''.join([PATH_RESULTS, '/all_data', '_csv']), 'wb') as file:
    pickle.dump(all_data, file)
