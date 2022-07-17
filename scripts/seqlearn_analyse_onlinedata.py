'''First pass analysing behavioral data'''

#%%
import os
from collections import Counter
import pickle
import numpy as np
import pandas as pd
import src.behavior_preprocess as prep

#%% SCRIPT PARAMETERS
PATH_DATA = "data/v4"
PATH_DEMOGRAPHIC = "data/prolific/demographics"
PATH_RESULTS = "data/v4/preprocessed"

NB_BLOCK_FAR = 14 #index of far transfer block
NB_BLOCK_1STEP = 13

#%%GET ALL DATA INTO DATAFRAME

df_subject = prep.fetch_demographics(PATH_DEMOGRAPHIC, PATH_DATA)

with open('data_log.txt', "r", encoding="utf8") as file:
    log = file.read().split('\n')
df_log = pd.DataFrame([i.split(' ')[1:] for i in log])

all_data = []
for file in os.listdir(PATH_DATA):
    #if df_log[df_log.iloc[:, -1] == file][8].values[0] != 'Jun':
    #    continue

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

with open('data_log.txt', "r", encoding="utf8") as file:
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
                print(f'{file} completed at {time[0]} on {day[0]}')

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
with open('bonus.csv', 'w', encoding="utf8") as out:
    datasel = all_data[(all_data.expt_group == 'complex') & (all_data.expt_curriculum == 'blocked_magician')]
    for subj, subj_data in datasel.groupby('expt_subject'):
        #if int(df_log[df_log.iloc[:, -1] == subj + '.txt'][9].values[0]) < 29:
        #    continue
        prolific_id = subj_data['expt_turker'].unique()
        bonus_vec = np.array(subj_data['block_bonus'].iloc[0])
        bonus_vec = bonus_vec[bonus_vec != np.array(None)]
        bonus = np.round(sum([bonus for bonus in bonus_vec if bonus > 0]), decimals=2)

        group = subj_data['expt_group'].unique()[0]
        curr = subj_data['expt_curriculum'].unique()[0]
        subid = subj_data['expt_subject'].unique()[0]
        out.write(f'{prolific_id[0]},{bonus}\n')
        #out.write('{4} {2}-{3} {0},{1}\n'.format(subj, bonus, group, curr, subid))

#%% Preprocessing

# add info time-outs
timeouts = all_data.groupby(['expt_turker'], as_index=False)['resp_correct'].apply(
    lambda x: pd.Series({'n_timeout':x.isnull().sum()}))

all_data = pd.merge(all_data,timeouts, on='expt_turker')

df_acc = all_data.groupby(['expt_turker', 'expt_block'], as_index=False)['resp_correct'].apply(
    lambda x: pd.Series({'acc':x.sum()/x.count()}))

# add info on far transfer
df_acc['generaliser'] = False
for subj in df_acc.loc[(df_acc.acc > 0.6) & (df_acc.expt_block == NB_BLOCK_FAR),'expt_turker']:
    df_acc.loc[df_acc.expt_turker == subj,'generaliser'] = True
all_data = pd.merge(all_data,df_acc.loc[df_acc.expt_block==NB_BLOCK_FAR][['expt_turker','generaliser']],
                     on='expt_turker')

#add info any learning
df_learner = df_acc.groupby(['expt_turker'], as_index=False).apply(
            lambda x: pd.Series({'max_training_score': np.max(x['acc'].values[:7]) }))
all_data = pd.merge(all_data,df_learner, on='expt_turker')

#add info 1-step rule
step1_acc = all_data.loc[(all_data.expt_block==NB_BLOCK_1STEP) & (all_data.rule==3)].groupby(
            ['expt_turker'], as_index=False)['resp_correct'].apply(
            lambda x: pd.Series({'ruleF_acc': np.nanmean(x) }))
all_data = pd.merge(all_data, step1_acc, on='expt_turker')
step1_acc = all_data.loc[(all_data.expt_block==NB_BLOCK_1STEP) & (all_data.rule==5)].groupby(
            ['expt_turker'], as_index=False)['resp_correct'].apply(
            lambda x: pd.Series({'ruleF_acc': np.nanmean(x) }))
all_data = pd.merge(all_data, step1_acc, on='expt_turker')

#%% Save data

with open(''.join([PATH_RESULTS, '/all_data', '_csv']), 'wb') as file:
    pickle.dump(all_data, file)
