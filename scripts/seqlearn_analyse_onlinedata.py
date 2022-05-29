'''First pass analysing behavioral data'''

#%%
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.behavior_preprocess as prep

#%% GET ALL DATA INTO DATAFRAME
PATH_DATA = "data/v3/data"
PATH_DEMOGRAPHIC = "data/prolific/demographics"
PATH_RESULTS = ""

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

with open(''.join([PATH_DATA, '/all_data', '_csv']), 'wb') as file:
    pickle.dump(all_data, file)

#%% ACCURACY
#extract correct/incorrect per participant & block
col_group = ['expt_turker', 'expt_block', 'expt_index']
np_acc = prep.pd2np(all_data, col_group)
np.nanmean(np_acc, axis=2)

#%%

#%%
##############################################
# First pass, quick stats for each submission, quality check
all_subj = list(set(all_data['expt_turker']))
subj = all_subj[0]
for subj in all_subj:
    print(subj)
    prep.output_submission_details(all_data, subj)

#%% Save bonus to file
with open('bonus.csv', 'w') as out:
    all_data.keys()
    for subj, subj_data in all_data.groupby('expt_turker'):
        bonus_vec = np.array(subj_data['block_bonus'].iloc[0])
        bonus_vec = bonus_vec[bonus_vec != np.array(None)]
        bonus = sum([bonus for bonus in bonus_vec if bonus > 0])

        group = subj_data['expt_group'].unique()[0]
        curr = subj_data['expt_curriculum'].unique()[0]
        subid = subj_data['expt_subject'].unique()[0]
        out.write('{4} {2}-{3} {0},{1}\n'.format(subj, bonus, group, curr, subid))
