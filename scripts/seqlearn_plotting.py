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

#%%
all_data.keys()
cols_plot = ['expt_group', 'expt_block', 'expt_curriculum', 'expt_turker']
df_acc = all_data.groupby(cols_plot)['resp_correct'].apply(lambda x: (np.nansum(x)/len(x))*100)
df_acc = pd.DataFrame(df_acc)

fig = plt.figure(figsize=(10, 10))
g = sns.lineplot(x='expt_block', y='resp_correct',
                 style='expt_curriculum', hue='expt_group', data=df_acc,
                 err_style='bars', err_kws={'capsize':6}, marker='o', ci=95)
plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
plt.title('main effect accuracy learning')
#plt.savefig('results/prolific/group/2step'+'main_acc.jpg', bbox_inches="tight")

#%%
# create an empty dictionary
test_results = {}
for i_block in set(df_acc.block_num):
    group1 = df_acc.where((df_acc.curriculum == 'interleaved') & (df_acc.block_num == i_block)).dropna()
    group2 = df_acc.where((df_acc.curriculum == 'magician') & (df_acc.block_num == i_block)).dropna()
    # add the output to the dictionary
    test_results[str(i_block)] = mannwhitneyu(group1['acc'], group2['acc'])
