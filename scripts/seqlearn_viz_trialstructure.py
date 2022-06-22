'''Visualize input structure'''

#%%
import itertools
import pickle
import operator
import numpy as np
import pandas as pd
from numpy import matlib
import matplotlib as mplt
import matplotlib.pyplot as plt
import src.my_py_utils as util

#%%
PATH_DATA = "data/v3/preprocessed"

with open(''.join([PATH_DATA, '/all_data', '_csv']), 'rb') as file:
    all_data = pickle.load(file)

subjs = all_data[all_data['expt_group'] == 'simple']['expt_subject'].unique()
n_blocks = len(all_data['expt_block'].unique())

with open('data_log.txt', "r") as file:
    log = file.read().split('\n')
df_log = pd.DataFrame([i.split(' ')[1:] for i in log])

#%% Plot trial structure per participant

# for each subject, get order of trials based on indices in unique sequence list
subj = subjs[0]
for subj in subjs:
    if subj != 'jK24bbd7lSAM':
    #if int(df_log[df_log.iloc[:, -1] == subj + '.txt'][9].values[0]) < 27:
        continue
    # get all unique sequences & map individual trial order re index in unique list of seqs
    unique_seq = []
    iloc = all_data.index[all_data['expt_subject'] == subj].tolist()[0]
    for seq_block in all_data['sequences'][iloc]:
        for seq in seq_block:
            seq = list(util.list2flat(seq))
            if seq not in unique_seq:
                unique_seq.append(seq)

    print(subj)
    subj_data = all_data[all_data.eq(subj).any(1)]

    all_trl_idx = []
    trialstruct = []
    MAX_IDX = 0
    for i_block, block_data in subj_data.groupby('expt_block'):
        sequences = block_data['seq']
        flat_sequences = sequences.apply(lambda x: list(util.list2flat(x)))

        unique_idx = []
        for seq in flat_sequences:
            for idx, u_seq in enumerate(unique_seq):
                if u_seq == seq:
                    unique_idx.append(idx)
        MAX_IDX = max(MAX_IDX, max(unique_idx))

        trial_num, n_step = np.vstack(block_data['magician']).shape
        steps = np.matlib.repmat(list(range(n_step)), 1, trial_num).flatten()
        magician = np.vstack(block_data['magician']).flatten()
        if n_step == 2:
            f = operator.itemgetter(1, 3)
        else:
            f = operator.itemgetter(1)
        rule = np.vstack(block_data['seq'].apply(lambda x: f(list(util.list2flat(x))))).flatten()
        ingredient = np.vstack(block_data['ingredient']).flatten()
        state = np.repeat(np.vstack(block_data['state']).flatten(), n_step)

        all_trl_idx.append(unique_idx)
        trialstruct.append(np.vstack([steps, rule, ingredient, state]))

    trialstruct = np.column_stack((itertools.zip_longest(*trialstruct, fillvalue=np.nan)))

    mplt.rc('font', size=50)
    #plot input structure
    n_blocks, _ = trialstruct.shape
    mymap = plt.cm.get_cmap('tab20', 16)
    fig, axes = plt.subplots(nrows=n_blocks, ncols=1, figsize=(60, 80))
    for i, (ax, struct) in enumerate(zip(axes.flat, trialstruct)):
        im = ax.imshow(np.vstack(struct)[:,1:401], cmap=mymap, vmin=0, vmax=15)
        ax.set_aspect('auto')
        ax.set_xticks(list(range(0,400,31)))
    fig.colorbar(im, ax=axes.ravel().tolist())

    #plot unique trial index
    fig = plt.figure(figsize=(60, 80))
    mymap = plt.cm.get_cmap('magma', 20)
    trl_idx_matrix = np.column_stack((itertools.zip_longest(*all_trl_idx, fillvalue=np.nan)))

    plt.imshow(trl_idx_matrix, cmap=mymap, vmin=0, vmax=MAX_IDX)
    plt.colorbar()
    plt.title(subj)
