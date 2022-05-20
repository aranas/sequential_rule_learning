'''First pass analysing behavioral data'''

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.behavior_preprocess as prep

#%% GET ALL DATA INTO DATAFRAME
PATH_DATA = "data/prolific/data/2step_V2"
PATH_DEMOGRAPHIC = "data/prolific/demographics"
PATH_RESULTS = "results/prolific/2step_V2/"

df_subject = prep.fetch_demographics(PATH_DEMOGRAPHIC, PATH_DATA)

all_data = []
for file in os.listdir(PATH_DATA):
    PATH_TO_FILE = '/'.join([PATH_DATA, file])
    if not PATH_TO_FILE.endswith('.txt'):
        continue
    #retrieve data and put into large pandas dataframe
    data_dicts = prep.data2dict(PATH_TO_FILE)
    all_data.append(prep.dicts2df(data_dicts))

all_data = pd.concat(all_data).reset_index()

#extract correct/incorrect per participant & block
col_group = ['edata-expt_turker', 'sdata-expt_block', 'sdata-expt_index']
np_acc = prep.pd2np(all_data, col_group)

##############################################
# Quick accuracy overview
df_acc = all_data[['edata-expt_turker', 'sdata-expt_block', 'sdata-resp_correct']].groupby(['edata-expt_turker', 'sdata-expt_block']).agg(['sum', 'count'])
df_acc.columns = ['_'.join(column) for column in df_acc.columns]
df_acc['acc'] = df_acc['sdata-resp_correct_sum']/df_acc['sdata-resp_correct_count']
df_acc = df_acc['acc'].unstack()

# First pass, quick stats for each submission, quality check
for filename in  df_subject['filename'].values:
    #if 'simple' in  set(df_data[df_data['i_subject'] == filename]['group']):
    #    continue
    #if 'magician' not in set(df_data[df_data['i_subject'] == filename]['curriculum']):
    #    continue
    print(filename)
    prep.output_submission_details(df_subject, filename)
    print('PERFORMANCE: ')
    for iblock in range(len(df_acc.columns)):
        print(np.round(df_acc[iblock][filename], decimals=2))

#%%
# Compute bonus
for file_name in df_subject['filename'].values:
    path_file = os.path.join(PATH_DATA, file_name)
    prolific_id = df_subject[df_subject['filename'] == file_name]['participant_id']
    recorded_bonus = df_subject[df_subject['filename'] == file_name]['bonus']

    single_data = df_data[df_data['i_subject'] == file_name]
    test_block_id = max(single_data['block_num'])
    n_blocks = len(set(single_data['block_num']))-1

    if 'simple' in  set(df_data[df_data['i_subject'] == file_name]['group']):
        continue
    if 'input' not in set(df_data[df_data['i_subject'] == file_name]['curriculum']):
        continue

    total_bonus = 0
    for i_block in set(single_data['block_num']):
        #if i_block == test_block_id:
        #    continue
        response_correct =  single_data[single_data['block_num'] == i_block]['correct']
        _, bonus = prep.compute_bonus([0 if i is None else i for i in response_correct], 3/n_blocks)
        if bonus > 0:
            total_bonus = total_bonus+bonus

    print('{0},{1}'.format(prolific_id.values[0], recorded_bonus.values[0]))
    #print(file_name)
    #print(total_bonus)

#%% Visualize trial order, why did some people learn so well?
all_reps = []
for filename in  df_subject['filename'].values:
    print(filename)
    dt_tmp = df_data[df_data['i_subject'] == filename]

    f = os.path.join(PATH_DATA, filename)
    [s_data, _, parameters_data] = prep.retrieve_data(f, ['sdata', 'edata', 'parameters'])
    print(s_data['bonus'])
    print(len(s_data['bonus']))
    count_rep_blocked = []
    for trials in parameters_data['block']['trialorder']:
        count_rep = 0
        for idx, seq_num in enumerate(trials[:-1]):
            if seq_num == trials[idx+1]:
                count_rep += 1
        count_rep_blocked.append(count_rep)
    print(count_rep_blocked)
    all_reps.append(count_rep_blocked)

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
        plt.ylim(y_min,y_max)
        plt.xlim(0,32)

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
# Draw a scatter plot
import seaborn as sns

df_select = df_acc[df_acc['group'] == 'complex'].copy()
df_select['block_num'] = df_select['block_num']+1
df_tmp = pd.concat([df_select, df_simple], axis=0)

plt.figure()
g = sns.lineplot(x='block_num', y='acc', style='curriculum', hue='group', data=df_tmp,
                  err_style='bars', err_kws={'capsize':6}, marker='o', ci=95)
plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
plt.title('main effect accuracy learning')
plt.savefig('results/prolific/group/2step'+'main_acc.jpg', bbox_inches="tight")

plt.figure()
g=sns.lineplot(x='block_num', y='rt', style='curriculum', hue='group', data=df_tmp[df_tmp['block_num'] != 11],
            err_style='bars', err_kws={'capsize':6}, marker='o', ci=95)
plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
plt.title('main effect RTs')
plt.savefig('results/prolific/group/2step'+'main_RT.jpg', bbox_inches="tight")

#%%
from scipy.stats import mannwhitneyu

import itertools
# create an empty dictionary
test_results = {}

for i_block in set(df_acc.block_num):
    group1 = df_acc.where((df_acc.curriculum == 'interleaved') & (df_acc.block_num == i_block)).dropna()
    group2 = df_acc.where((df_acc.curriculum == 'magician') & (df_acc.block_num == i_block)).dropna()
    # add the output to the dictionary
    test_results[str(i_block)] = mannwhitneyu(group1['acc'], group2['acc'])
