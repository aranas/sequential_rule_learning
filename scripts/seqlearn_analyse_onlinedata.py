'''First pass analysing behavioral data'''

#%%
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.behavior_preprocess as prep

#%% GET ALL DATA INTO DATAFRAME
PATH_DATA = "data/test"
PATH_DEMOGRAPHIC = "data/prolific/demographics"
PATH_RESULTS = ""

df_subject = prep.fetch_demographics(PATH_DEMOGRAPHIC, PATH_DATA)

all_data = []
for file in os.listdir(PATH_DATA):
    print(file)
    PATH_TO_FILE = '/'.join([PATH_DATA, file])
    if not PATH_TO_FILE.endswith('.txt'):
        continue
    #retrieve data and put into large pandas dataframe
    data_dicts = prep.data2dict(PATH_TO_FILE)
    data_dicts[0]
    all_data.append(prep.dicts2df(data_dicts))

all_data = pd.concat(all_data).reset_index(drop=True)
all_data = all_data.drop(0)
all_data.columns = [str.split('-',1)[1] for str in all_data.columns]

#extract correct/incorrect per participant & block
col_group = ['expt_turker', 'expt_block', 'expt_index']
np_acc = prep.pd2np(all_data, col_group)

np.nanmean(np_acc[0], axis=1)

#%%
##############################################

all_data.keys()
# First pass, quick stats for each submission, quality check
all_subj = list(set(all_data['expt_turker']))
for subj in all_subj:
    print(subj)
    prep.output_submission_details(all_data, subj)

#%% Plot trial structure per participant
def list2flat(alist):
  for item in alist:
    if isinstance(item, list):
      for subitem in item: yield subitem
    else:
      yield item

# get overview of unique sequences
unique_seq = []
for seq_block in all_data['sequences'][1]:
    for seq in seq_block:
        seq = list(list2flat(seq))
        if seq not in unique_seq:
            unique_seq.append(seq)

# for each subject, get order of trials based on indices in unique sequence list
all_trl_idx = []
for subj in all_subj:
    subj_data = all_data[all_data['expt_turker'] == subj]

    for i_block in all_data['expt_block'].unique():
        sequences = subj_data[subj_data['expt_block'] == i_block]['seq']
        flat_sequences = sequences.apply(lambda x: list(list2flat(x)))

        unique_idx = []
        for seq in flat_sequences:
            for idx, u_seq in enumerate(unique_seq):
                if u_seq == seq:
                    unique_idx.append(idx)
        all_trl_idx.append(unique_idx)
# viz trial structure
cols = ['magician', 'ingredient', 'state', 'resp_correct']
feedback = np.array(all_data['block-feedback'][1])

n_blocks = len(all_trl_idx)
trialstruct = []
for i_subj, subj in enumerate(all_subj):
    for i_block in range(n_blocks):
        tmp_df = all_data.loc[(all_data['expt_turker'] == subj) & (all_data['expt_block'] == i_block)]
        tmp_df = tmp_df[cols]

        trial_num, n_step = np.vstack(tmp_df['magician']).shape
        steps = np.matlib.repmat(list(range(n_step)), 1, trial_num).flatten()
        magician = np.vstack(tmp_df['magician']).flatten()
        ingredient = np.vstack(tmp_df['ingredient']).flatten()
        state = np.repeat(np.vstack(tmp_df['state']).flatten(), n_step)
        trlidx = np.repeat(all_trl_idx[i_block], n_step)

        trialstruct.append(np.vstack([steps, magician, ingredient, state, trlidx]))

trialstruct = np.column_stack((itertools.zip_longest(*trialstruct, fillvalue=np.nan)))

#%%
#create custom colormap
# map trial idx onto continuous colormap
# map categorical values (e.g. mag or ingredient id) onto other colors
import matplotlib.colors as mcolors
colors1 = plt.cm.get_cmap('tab20', 20)(np.linspace(0., 1, 16))
colors2 = plt.cm.get_cmap('cool', 20)(np.linspace(0, 1, 72))
# cmbine them and build a new colormap
colors = np.vstack((colors1, colors2))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

trialstruct[6]
fig, axes = plt.subplots(nrows=n_blocks+1, ncols=1, figsize=(30, 30))
for i, (ax, struct) in enumerate(zip(axes.flat, trialstruct)):
    tmp = struct.copy()
    tmp[4] = tmp[4]+15
    im = ax.imshow(np.vstack(tmp), cmap=mymap, vmin=0, vmax=72)
axes[-1].imshow(np.array([range(16)]), cmap=plt.cm.get_cmap('tab20', 16))
fig.colorbar(im, ax=axes.ravel().tolist())


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
