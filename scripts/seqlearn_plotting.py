'''Visualize results & run stats'''

import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu

#%%
PATH_DATA = "data/v3/data"

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
