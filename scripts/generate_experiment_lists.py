'''
This script generates stimulus lists for the prolific online experiment.
It saves the lists to txt files such they can be read in by the javascripts later on
Depending on the following parameters:

PATH_WRITE  - path where datafile is put
n_steps     - array of length n_blocks, contains length of sequences
rule_names  - array of length n_blocks,
            each entry in array contains vector specifying rule
inputs      - array of length n_blocks,
            each entry in array contains vector specifying input id
block_feedback - array, specifies whether feedback is shown for each block

It generate the following variables:
1) seqs - an array of sequences of shape: n_blocks x n_trials x NSTEPS+2, where
        n_blocks - # of blocks as specified through input parameters
        n_trials - # of sequences/trials
        and each sequence consists of a vector x, such that,
        x[0] - binary code for initial state.
        x[1:-1] - tuple with rule id in first position and binary input id in 2nd position.
        x[-1] - binary code for output state.
2) FIXME:DO I NEED THIS? codes - array of codes of length n_trials, with id per unique sequence.

variables are to file as a dict to 'PATH_WRITE/{NSTEPS}_{rule_names}.txt'
'''

# IMPORT STATEMENTS
from operator import itemgetter
import pickle
import numpy as np
import seqlearn_magicspell as magic

#%% SET PARAMETERS
PATH_WRITE = 'results/online_experiment/stimulus_lists/'
SUFFIX = '_C_test'
block_rule_names = [['forceB', 'crossB'], ['forceB', 'crossB'],
                    ['forceB', 'crossB'], ['forceB', 'crossB'],
                    ['forceB', 'crossB'],
                    ['forceB', 'crossB'],
                    ['forceB', 'crossB']]
block_inputs = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [12, 13]] # use 12 & 13 for non-random images
block_n_steps = [2, 2, 2, 2, 2, 1, 2]
block_feedback = [1, 1, 1, 1, 1, 0, 1]
block_randomise = [1, 1, 1, 1, 0, 0, 0]

curriculum_block_input = True
curriculum_block_mag = True


all_param = [PATH_WRITE, block_rule_names,
             block_inputs, block_n_steps, block_feedback, block_randomise,
             curriculum_block_input, curriculum_block_mag]

#save parameters to file
with open(''.join([PATH_WRITE, 'parameters', SUFFIX]), 'wb') as file:
    pickle.dump(all_param, file)


#%% GENERATE VARS based on parameters & generate sequences
if len(set([len(block_n_steps), len(block_rule_names), len(block_inputs)])) > 1:
    print('number of blocks inconsistent across parameters')
    N_BLCK = 0
else:
    N_BLCK = len(block_n_steps)

uniq_rule_names = list({item for sublist in block_rule_names for item in sublist})
all_rule_names = list(map(itemgetter('name'), magic.dRules))
#all_operator_ids = np.array(range(len(uniq_rule_names)))
#rules = [x for x in magic.dRules if x['name'] in block_rule_names]

PATH_TO_OUTFILE = '/'.join([PATH_WRITE,
                            '{0}_{1}_{2}.js'.format(SUFFIX, N_BLCK, '_'.join(uniq_rule_names))])

#%% GENERATE SEQS
seqs = []
block_rule_ids = []
for n_steps, rule_names, inputs in zip(block_n_steps, block_rule_names, block_inputs):
    op = [all_rule_names.index(rule) for rule in rule_names]
    block_rule_ids.append(op)
    trials = magic.generate_trial(op, inputs, n_steps, replacement=True, for_js_script=True)
    out = magic.transform(trials, magic.dRules)

    seqs.append(np.vstack((np.array(trials).T, np.array(out))).T.tolist())

#%%
# apply additional conditions (e.g. keep only certain trials for blocking)
# duplicate trials within block if wanted
# will be applied to first 4 blocks
if curriculum_block_input | curriculum_block_mag:

    n_trainblocks = 4
    input_ids = list(set(np.array(block_inputs)[:n_trainblocks].flatten()))
    input_mag = sorted([all_rule_names.index(rule) for rule in uniq_rule_names])
    new_seqs = []
    for cnt, bseq in enumerate(seqs):
        if cnt >= 4:
            new_seqs.append(bseq)
            continue
        if curriculum_block_input:
            # first trials that start with in1 and continue with in1
            if cnt == 0:
                sel = [iseq for iseq in bseq if (iseq[1][1] == input_ids[1]) & (iseq[2][1] == input_ids[1])]
            if cnt == 1:
                sel = [iseq for iseq in bseq if (iseq[1][1] == input_ids[0]) & (iseq[2][1] == input_ids[0])]
            if cnt == 2:
                sel = [iseq for iseq in bseq if (iseq[1][1] == input_ids[0]) & (iseq[2][1] == input_ids[1])]
            if cnt == 3:
                sel = [iseq for iseq in bseq if (iseq[1][1] == input_ids[1]) & (iseq[2][1] == input_ids[0])]
            #sel = np.vstack((sel, sel, sel, sel)).tolist()
            new_seqs.append(sel)
        elif curriculum_block_mag:
            # first trials that start with in1 and continue with in1
            if cnt == 0:
                sel = [iseq for iseq in bseq if (iseq[1][0] == input_mag[0]) & (iseq[2][0] == input_mag[0])]
            if cnt == 1:
                sel = [iseq for iseq in bseq if (iseq[1][0] == input_mag[1]) & (iseq[2][0] == input_mag[1])]
            if cnt == 2:
                sel = [iseq for iseq in bseq if (iseq[1][0] == input_mag[1]) & (iseq[2][0] == input_mag[0])]
            if cnt == 3:
                sel = [iseq for iseq in bseq if (iseq[1][0] == input_mag[0]) & (iseq[2][0] == input_mag[1])]
            #sel = np.vstack((sel, sel, sel, sel)).tolist()
            new_seqs.append(sel)

    seqs = new_seqs

#%% compute total trials
N_TOTAL_TRLS = 0
nb_trl_block = []
for bseq in seqs:
    nb_trl_block.append(len(bseq))
    N_TOTAL_TRLS = N_TOTAL_TRLS+len(bseq)

#%% SAVE DATA
with open(PATH_TO_OUTFILE, 'w') as f:
    f.write('parameters.total_trials = {0};\n'.format(N_TOTAL_TRLS))
    f.write('parameters.block.ruleNames = {0};\n'.format(block_rule_names))
    f.write('parameters.block.ruleID = {0};\n'.format(block_rule_ids))
    f.write('parameters.block.inputID = {0};\n'.format(block_inputs))
    f.write('parameters.block.feedback = {0};\n'.format(block_feedback))
    f.write('parameters.block.randomise = {0};\n'.format(block_randomise))
    f.write('parameters.block.nb_trials = {0};\n'.format(nb_trl_block))

    f.write('parameters.sequences = ')
    f.write('[')
    for n_blk, block in enumerate(seqs):
        f.write('[')
        n_trls = len(block)
        for i, elem in enumerate(block):
            if i+1 == n_trls:
                f.write('%s]' % elem)
            else:
                f.write('%s,\n' % elem)
        if n_blk != N_BLCK-1:
            f.write(',\n')
    f.write('];')
