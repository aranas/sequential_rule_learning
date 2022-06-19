'''Visualize results & run stats'''

import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu
import src.behavior_preprocess as prep

#%%
PATH_DATA = "data/v3/preprocessed"

with open(''.join([PATH_DATA, '/all_data', '_csv']), 'rb') as file:
    all_data = pickle.load(file)

subjs = all_data['expt_subject'].unique()
n_blocks = len(all_data['expt_block'].unique())

#%% fix condition naming; exclude single step block; exclude practice trial (block == nan)
all_data.loc[all_data['expt_group'] == 'simple', 'expt_curriculum'] = 'simple_blocked_magician'
all_data = all_data[all_data.expt_block != 9]
all_data = all_data[~all_data['expt_block'].isnull()]
#%% subselect data
#cond1 = (all_data.expt_group == 'simple') & (all_data.expt_curriculum == 'blocked_input')
cond2 = (all_data.expt_group == 'simple')
cond3 = (all_data.expt_group == 'complex') & (all_data.expt_curriculum == 'interleaved')
all_data = all_data.loc[cond2 | cond3]

cols_plot = ['expt_group', 'expt_block', 'expt_curriculum', 'expt_turker']
df_acc = all_data.groupby(cols_plot, as_index=False)['resp_correct'].apply(lambda x: (np.nansum(x)/len(x))*100)
df_acc = pd.DataFrame(df_acc)
#as_index=False

#%%
fig = plt.figure(figsize=(20, 10))
g = sns.lineplot(x='expt_block', y='resp_correct',
                 style='generaliser',
                 hue='expt_curriculum',
                 data=df_acc,
                 err_style='bars', err_kws={'capsize':6}, marker='o', ci=95)
plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
plt.title('main effect accuracy learning')

fig = plt.figure(figsize=(10, 10))
ax = sns.swarmplot(x='expt_block', y='resp_correct', hue='expt_curriculum', data=df_acc)
plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)

plt.savefig('results/prolific/v3/main_acc_complex_leavvsinput.jpg', bbox_inches="tight")

#%% Mann-Whitney-Test
all_data['expt_curriculum'].unique()
# create an empty dictionary
test_results = {}
for i_block in all_data['expt_block'].unique():
    group1 = df_acc.loc[(df_acc.expt_curriculum == 'blocked_magician') & (df_acc.generaliser == True) & (df_acc.expt_block == i_block)]
    group2 = df_acc.loc[(df_acc.expt_curriculum == 'blocked_magician') & (df_acc.generaliser == False) & (df_acc.expt_block == i_block)]
    # add the output to the dictionary
    test_results[str(i_block)] = mannwhitneyu(group1['resp_correct'], group2['resp_correct'])

#%% Exclude participants
timeout_thres = 10
n_many_timeouts = all_data[all_data['n_timeout'] >= timeout_thres]['expt_subject'].nunique()
print('exclude {0} - too many time-outs'.format(n_many_timeouts))
all_data = all_data[all_data['n_timeout'] <= timeout_thres]

# include only ppl that learned
n_no_learning = all_data[~all_data['learned_1'] | ~all_data['learned_2'] | ~all_data['learned_3'] | ~all_data['learned_4']]['expt_subject'].nunique()
all_data = all_data.loc[all_data['learned_1'] & all_data['learned_2'] & all_data['learned_3'] & all_data['learned_4']]
print('exclude {0} - not learning throughout blocks '.format(n_no_learning))

col_group = ['expt_group', 'expt_curriculum', 'expt_block']
all_data.groupby(col_group)['expt_subject'].nunique()


#%% Plot individual subject data (RT, time-outs, moving average)
# select data
col_group = ['expt_turker','expt_index']
np_acc = prep.pd2np(all_data, col_group, 'resp_correct')
nsubj, n_ttrials = np_acc.shape

#np_rt = prep.pd2np(all_data, col_group, 'resp_reactiontime')
#nsubj, n_ttrials = np_rt.shape

# fetch modalities per participant
modalities = all_data.groupby(['expt_turker', 'expt_curriculum'],as_index=False).sum()['expt_curriculum'].to_numpy()
len(modalities)

#moving average (to capture progression within blocks as well)
window_size = 16
stride = 16
mov_avg = np.squeeze([[np.nanmean(np_acc[:,i:i+window_size],axis=1)]for i in range(0, n_ttrials, stride)
                   if i+window_size <= n_ttrials]).T
mov_avg.shape

#fig = plt.figure(figsize=(10,10))
#plt.imshow(mov_avg.T)
uniq_modalities = list(set(modalities))

fig, axs = plt.subplots(len(uniq_modalities), 1, figsize=(15, 12), dpi=300, facecolor='w')
for i, (ax, modality) in enumerate(zip(axs, uniq_modalities)):

    # Select subjects for this modality
    idx = modality == modalities

    m_subj = mov_avg[idx,:]
    m = np.nanmean(m_subj, axis=0)
    se = np.nanstd(m_subj, axis=0)/np.sqrt(idx.sum())

    center_idx = np.array(range(0,len(m)))

    ax.errorbar(
            x = center_idx,
            y = m,
            yerr = 2 * se,
            markersize = 5,
            marker = 'D',
            alpha = 1
        )

      # Plot individual data points
    for k in range(idx.sum()):
        ax.errorbar(
            x = center_idx + np.random.normal(0, 0.1, center_idx.size),
            y = m_subj[k],
            markersize = 3,
            marker = 'o',
            alpha = .1,
            lw = 0.3, # Do not plot the lines (this is messy)
        )
     # Aesthetics
    ax.set_title(modality, fontweight='bold')
    ax.set_xlabel('Trial #')
    ax.set_ylabel('Accuracy')
    #ax.set_ylim(5, 7) #for RTs

    ax.axvline(4, color='k', ls='-', alpha=.4)
    ax.axvline(8, color='k', ls='-', alpha=.4)
    ax.axvline(12, color='k', ls='-', alpha=.4)
    ax.axvline(16, color='k', ls='-', alpha=.4)
    ax.axvline(18, color='k', ls='-', alpha=.4)


    ax.axhline(1/2., color='k', ls='--', alpha=.4)

plt.tight_layout()
