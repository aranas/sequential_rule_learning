'''First pass analysing behavioral data'''

#%%
import os
import itertools
import numpy as np
import pandas as pd
import pickle
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
all_data.keys()
all_data['debrief_magician']
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
        bonus = sum([bonus for bonus in bonus_vec if bonus>0])

        group = subj_data['expt_group'].unique()[0]
        curr = subj_data['expt_curriculum'].unique()[0]
        subid = subj_data['expt_subject'].unique()[0]
        out.write('{4} {2}-{3} {0},{1}\n'.format(subj, bonus, group, curr,subid))

#%% Plot data per participant
for file_name in df_subject['filename'].values:
    path_file = os.path.join(PATH_DATA, file_name)
    #extract some variables from the data
    [s_data, e_data, p_data] = prep.retrieve_data(path_file, ['sdata', 'edata', 'parameters'])
    trial_duration = p_data['timing']['seqduration']/1000

    single_data = df_data[df_data['i_subject'] == file_name]

    n_trials_presented = len(single_data)
    n_block = len(np.unique(single_data['block_num']))

    #Visualize performance

    fig = plt.figure(figsize=(30, 15))
    for iblock in np.unique(single_data['block_num']):
        if iblock == 10:
            break
        blocked_rt = single_data[single_data['block_num'] == iblock]['rt']
        idx_timeouts = [i for i, value in enumerate(blocked_rt) if value is None]
        blocked_correct = single_data[single_data['block_num'] == iblock]['correct']
        idx_incorrect = np.where(blocked_correct != 1)
        idx_unique = prep.retrieve_uniqueness_point(single_data['seqid'])

        trials_arr = list(range(len(blocked_rt)))
        # set time-outs to 0 RT and subtract duration of trial from response counter
        blocked_rt = [0 if i is None else i-trial_duration for i in blocked_rt]
        y_min = np.min(blocked_rt)
        y_max = np.max(blocked_rt)

        idx_incorrect = [i for i in idx_incorrect[0] if i not in idx_timeouts]

        # plot RTs
        plt.subplot(n_block, 1, iblock+1)
        plt.plot(trials_arr, blocked_rt)
        plt.vlines(idx_incorrect, y_min, y_max, 'k')
        plt.vlines(idx_timeouts, y_min, y_max, 'r')
        plt.vlines(idx_unique, y_min, y_max, 'b')
        plt.ylim(y_min, y_max)
        plt.xlim(0, 32)

    fig.legend(['reaction time', 'incorrect trials', 'time_out', 'uniqueness point'], loc='lower center')
    fig.suptitle('subject {0} - group {1}'.format(set(single_data['i_subject']), set(single_data['group'])))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.savefig(PATH_RESULTS +'summaryfig_' + file_name[:-4] + '.jpg')

#%%
# Visualize inferred rules (from test block no feedback)
all_rule_hat = np.zeros((1,8))
for file_name in df_subject['filename'].values:
    path_file = os.path.join(PATH_DATA, file_name)
    if 'complex' in  set(df_data[df_data['i_subject'] == file_name]['group']):
        continue
    if 'interleaved' not in set(df_data[df_data['i_subject'] == file_name]['curriculum']):
        continue
    print(file_name)

    [trial_data, e_data, p_data] = prep.retrieve_data(path_file, ['sdata', 'edata', 'parameters'])

    n_testtrials = len(p_data['block']['ruleID'][-1])
    rule_hat = np.array(trial_data['resp_category'][-n_testtrials*4:])
    rule = np.array(trial_data['target_response'][-n_testtrials*4:])

    all_rule_hat = all_rule_hat + rule_hat

plt.figure()
    # state-input order per trial, rules are blocked: 0-0, 0-1, 1-0, 1-1
tmp_count = 0
print('Ingredient # {0}'.format(p_data['block']['inputID'][-4]))
for i in range(n_testtrials):
    print('Magician # {0}'.format(p_data['block']['magID'][-1][i]))
    tmp_count += 1
    tmp_idx = 4*i
    plt.subplot(n_testtrials, 2, tmp_count)
    plt.imshow(rule[tmp_idx:tmp_idx+4].reshape(2, 2))
    tmp_count += 1
    plt.subplot(n_testtrials, 2, tmp_count)
    plt.imshow(all_rule_hat.T[tmp_idx:tmp_idx+4].reshape(2, 2))
plt.colorbar()
plt.suptitle('complex-input')
plt.savefig(PATH_RESULTS +'input.jpg')


#%%
## subject-level metric
df_acc = pd.DataFrame(columns=['i_subject', 'group', 'curriculum', 'block_num', 'acc', 'rt'])
for file_name in df_subject['filename'].values:
    for i_block in set(df_data.block_num):

        df_tmp = df_data[(df_data['i_subject'] == file_name) & (df_data['block_num'] == i_block)]

        row = pd.Series([
            file_name,
            df_tmp.group.unique()[0],
            df_tmp.curriculum.unique()[0],
            i_block,
            df_tmp['correct'].mean()*100,
            df_tmp['rt'].mean()
        ], index=df_acc.columns)

        df_acc = df_acc.append(row, ignore_index=True)

mean_acc = df_acc.groupby(['block_num', 'group', 'curriculum']).agg(['mean', np.nanstd, 'count'])

#%%
