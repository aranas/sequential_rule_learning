'''First pass analysing behavioral data'''

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.behavior_preprocess as prep

#%% GET ALL DATA INTO DATAFRAME
path_data = "data/prolific/data/v3"
path_demographic = "data/prolific/demographics"
path_results = 'results/prolific/2step_v1/'

df_subject = prep.fetch_demographics(path_demographic, path_data)
df_data = prep.fetch_data(path_data, df_subject['filename'].values)

# Quick accuracy overview
df_acc = df_data[['i_subject', 'block_num', 'correct']].groupby(['i_subject', 'block_num']).agg(['sum', 'count'])
df_acc.columns = ['_'.join(column) for column in df_acc.columns]
df_acc['acc'] = df_acc['correct_sum']/df_acc['correct_count']
df_acc = df_acc['acc'].unstack()

# First pass, quick stats for each submission, quality check
for filename in  df_subject['filename'].values:
    start_date = df_subject[df_subject['filename'] == filename]['started_datetime']
    #if not start_date.str.startswith('2022-03-30').values[0]:
    #    continue

    print(filename)
    prep.output_submission_details(df_subject, filename)
    print('PERFORMANCE: ')
    for iblock in range(len(df_acc.columns)):
        print(np.round(df_acc[iblock][filename], decimals=2))

# Compute bonus
for file_name in df_subject['filename'].values:
    path_file = os.path.join(path_data, file_name)
    prolific_id = df_subject[df_subject['filename'] == file_name]['participant_id']
    single_data = df_data[df_data['i_subject'] == file_name]
    test_block_id = max(single_data['block_num'])
    n_blocks = len(set(single_data['block_num']))-1

    total_bonus = 0
    for i_block in set(single_data['block_num']):
        if i_block == test_block_id:
            continue
        response_correct =  single_data[single_data['block_num'] == i_block]['correct']
        _, bonus = prep.compute_bonus([0 if i is None else i for i in response_correct], 3/n_blocks)
        if bonus > 0:
            total_bonus = total_bonus+bonus

    print(prolific_id)
    print(file_name)
    print(total_bonus)

#%% Visualize trial order, why did some people learn so well?
all_reps = []
for filename in  df_subject['filename'].values:
    print(filename)
    dt_tmp = df_data[df_data['i_subject'] == filename]

    f = os.path.join(path_data, filename)
    [_, _, parameters_data] = prep.retrieve_data(f, ['sdata','edata','parameters'])

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
    path_file = os.path.join(path_data, file_name)

    #extract some variables from the data
    [s_data, e_data, p_data] = prep.retrieve_data(path_file, ['sdata', 'edata', 'parameters'])
    trial_duration = p_data['timing']['seqduration']/1000

    single_data = df_data[df_data['i_subject'] == file_name]

    n_trials_presented = len(single_data)
    n_block = len(np.unique(single_data['block_num']))

    #Visualize performance

    fig = plt.figure(figsize=(30, 15))
    for iblock in np.unique(single_data['block_num']):
        if iblock == 11:
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

    fig.legend(['reaction time', 'incorrect trials', 'time_out', 'uniqueness point'], loc='lower center')
    fig.suptitle('subject {0} - group {1}'.format(set(single_data['i_subject']), set(single_data['group'])))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.savefig(path_results +'summaryfig_' + file_name[:-4] + '.jpg')

#%%
# Visualize inferred rules (from test block no feedback)
for file_name in df_subject['filename'].values:
    path_file = os.path.join(path_data, file_name)
    [_, e_data, p_data] = prep.retrieve_data(path_file, ['sdata', 'edata', 'parameters'])

    n_testtrials = len(parameters_data['ruleid'][-1])
    print(n_testtrials)
    rule_hat = np.array(trial_data['resp_category'][-n_testtrials*4:])
    rule = np.array(trial_data['target_response'][-n_testtrials*4:])

    plt.figure()
    # state-input order per trial, rules are blocked: 0-0, 0-1, 1-0, 1-1
    tmp_count = 0
    for i in range(n_testtrials):
        tmp_count += 1
        tmp_idx = 4*i
        plt.subplot(n_testtrials, 2, tmp_count)
        plt.imshow(rule[tmp_idx:tmp_idx+4].reshape(2, 2))
        tmp_count += 1
        plt.subplot(n_testtrials, 2, tmp_count)
        plt.imshow(rule_hat[tmp_idx:tmp_idx+4].reshape(2, 2))

#%%

## Group Analysis pipeline
df_data = pd.DataFrame(columns=['i_subject','group',
                                'block_num', 'rule','acc', 'mean_RT'])
for isub, file_name in enumerate(df_out['filename'].values):
    f = os.path.join(DATA_DIR, file_name)
    data = prep.retrieve_data(f)

    nblock = parameters_data['nb_blocks']
    trial_data = data[0]['sdata']
    experiment_data = data[1]['edata']
    parameters_data = data[2]['parameters']

    ruleid, rulename = map_rule2ruleid(parameters_data['ruleid'])

    # preprocess
    n_testtrials = len(parameters_data['ruleid'][-1])
    nblock = parameters_data['nb_blocks']-n_testblock
    if n_testblock is 1 :
        all_correct = trial_data['resp_correct'][1:-n_testtrials*4]
    else:
        all_correct = trial_data['resp_correct'][1:]

    blocked_correct = np.array_split(all_correct, nblock)
    blocked_rt      = np.array_split(trial_data['resp_reactiontime'][1:], nblock)

    acc_blocked = []
    rt_blocked = []
    count_rep_blocked = []
    for iblock, block_data in enumerate(blocked_correct):

        #num of repeating adjacent trials
        trialorder = parameters_data['block']['trialorder'][iblock]
        count_rep = 0
        for idx, seq_num in enumerate(trialorder[:-1]):
            if seq_num == trialorder[idx+1]:
                count_rep += 1
        count_rep_blocked.append(count_rep)
        #FIXME: maybe rather want to treat timeouts as incorrect?
        acc_blocked.append(pd.Series(block_data).mean())
        rt_blocked.append(pd.Series(blocked_rt[iblock]).mean())

    num_timeouts = len([i for i in range(len(trial_data['resp_reactiontime'])) if trial_data['resp_reactiontime'][i] == None])
    if num_timeouts > 4:
        #print('exclude subject {0}from group {1}'.format(file_name,experiment_data['expt_group']))
        print('timeouts {0}'.format(str(num_timeouts)))
        #print(acc_blocked)
        #continue

    for iblock, block_data in enumerate(blocked_correct):
        block_name ='_'.join(['block',str(iblock)])
        row = pd.Series([
            file_name,
            experiment_data['expt_group'],
            block_name,
            parameters_data['ruleid'][iblock],
            acc_blocked[iblock],
            rt_blocked[iblock],
        ], index=df_data.columns)
        df_data = df_data.append(row, ignore_index=True)

mean_acc = df_data.groupby(['block_num','group']).agg(['mean',np.nanstd,'count'])
df_data.replace('block_0','1',inplace=True)
df_data.replace('block_1','2',inplace=True)
df_data.replace('block_2','3',inplace=True)
df_data.replace('block_3','4',inplace=True)

#%%
# Draw a scatter plot
import seaborn as sns

plt.figure()
g = sns.catplot(x='block_num',y='acc', hue='group',data=df_data, kind="box")
g.map_dataframe(sns.stripplot, x="block_num", y="acc",
                hue="group", alpha=0.6, dodge=True)

plt.figure()
g = sns.lineplot(x='block_num',y='acc', hue='group',data=df_data, err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('main effect accuracy learning')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc.jpg',bbox_inches="tight")

plt.figure()
g=sns.lineplot(x='block_num',y='acc_corrected', hue='group',data=df_plot, err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('main effect accuracy - corrected for reps')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc_noreps.jpg',bbox_inches="tight")


plt.figure()
g=sns.lineplot(x='block_num',y='mean_RT', hue='group',data=df_plot, err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('main effect RTs - corrected for reps')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_RT_noreps.jpg',bbox_inches="tight")


plt.figure()
g=sns.lineplot(x='block_num',y='acc', style='last_block', hue='group',data=df_plot, err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('detailed effects acc')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc_detailed.jpg',bbox_inches="tight")

plt.figure()
g=sns.lineplot(x='block_num',y='acc', style='curriculum', data=df_plot[df_plot['group']=='control'], err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('Control group only - ACC')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc_detailed2.jpg',bbox_inches="tight")


plt.figure()
g=sns.lineplot(x='block_num',y='acc_corrected', style='last_block', hue='group',data=df_plot, err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('detailed effects acc - corrected')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc_detailed_corrected.jpg',bbox_inches="tight")

# Are some rule easier to learn than others?
plt.figure()
sns.stripplot(x='first_block',y='acc',data=df_plot[df_plot['block_num']=='block_0'],jitter=True)
plt.title('rule difficulty in 1st block')
plt.savefig('results/prolific/group/'+'rule_difficulty.jpg',bbox_inches="tight")

plt.figure()
sns.stripplot(x='first_block',y='acc_corrected',data=df_plot[df_plot['block_num']=='block_0'],jitter=True)
plt.title('rule difficulty in 1st block - corrected for reps')
plt.savefig('results/prolific/group/'+'rule_difficulty_corrected.jpg',bbox_inches="tight")

plt.figure()
g=sns.lineplot(x='block_num',y='acc_rest',  hue='lucky_guess',data=df_plot, err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(['1st incorrect','1st correct'],bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('main effect accuracy - divided by success on first trial')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'split_firsttrial.jpg',bbox_inches="tight")

#%%
from scipy.stats import mannwhitneyu

import itertools
# create an empty dictionary
test_results = {}

group1 = df_data.where((df_data.group == 'complex') & (df_data.block_num == 'block_5')).dropna()
group2 = df_data.where((df_data.group== 'simple') & (df_data.block_num == 'block_5')).dropna()
# add the output to the dictionary
test_results['main'] = mannwhitneyu(group1['acc_corrected'],group2['acc_corrected'])
