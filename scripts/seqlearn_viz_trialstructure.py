'''Visualize input structure'''

#%%
import itertools
import pickle
import operator
import numpy as np
import matplotlib.pyplot as plt
import src.my_py_utils as util

#%%
PATH_DATA = "data/v3/data/test"

with open(''.join([PATH_DATA, '/all_data', '_csv']), 'rb') as file:
    all_data = pickle.load(file)

subjs = all_data[all_data['expt_group'] == 'simple']['expt_turker'].unique()
n_blocks = len(all_data['expt_block'].unique())

#%% Plot trial structure per participant

# for each subject, get order of trials based on indices in unique sequence list
for subj in subjs:
    # get all unique sequences & map individual trial order re index in unique list of seqs
    unique_seq = []
    iloc = all_data.index[all_data['expt_turker'] == subj].tolist()[0]
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

    #plot input structure
    n_blocks, _ = trialstruct.shape
    mymap = plt.cm.get_cmap('tab20', 16)
    fig, axes = plt.subplots(nrows=n_blocks, ncols=1, figsize=(10, 10))
    for i, (ax, struct) in enumerate(zip(axes.flat, trialstruct)):
        im = ax.imshow(np.vstack(struct), cmap=mymap, vmin=0, vmax=15)
    fig.colorbar(im, ax=axes.ravel().tolist())


    #plot unique trial index
    fig = plt.figure(figsize=(10, 10))
    mymap = plt.cm.get_cmap('magma', 20)
    trl_idx_matrix = np.column_stack((itertools.zip_longest(*all_trl_idx, fillvalue=np.nan)))
    plt.imshow(trl_idx_matrix, cmap=mymap, vmin=0, vmax=MAX_IDX)
    plt.colorbar()
    plt.title(subj)
