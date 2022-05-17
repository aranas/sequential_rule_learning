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
import pickle
import numpy as np
import scripts.seqlearn_magicspell as magic

#%% SET PARAMETERS
PATH_WRITE = '.'
block_n_steps = [1]
block_rule_names = [['forceB', 'crossB', 'reverse']]
block_inputs = [[0, 1]]

#%% GENERATE VARS based on parameters & generate sequences
if len(set([len(block_n_steps), len(block_rule_names), len(block_inputs)])) > 1:
    print('number of blocks inconsistent across parameters')
    N_BLCK = 0
else:
    N_BLCK = len(block_n_steps)

all_rule_names = {item for sublist in block_rule_names for item in sublist}
rules = [x for x in magic.dRules if x['name'] in all_rule_names]
PATH_TO_OUTFILE = '/'.join([PATH_WRITE, '{0}_{1}.txt'.format(N_BLCK, '_'.join(all_rule_names))])
operators = list(range(len(all_rule_names)))

#%% GENERATE SEQS
seqs = []
for n_steps, rule_names, inputs in zip(block_n_steps, block_rule_names, block_inputs):

    trials = magic.generate_trial(operators, inputs, n_steps, replacement=True, for_js_script=True)
    out = magic.transform(trials, rules)
    seqs.append(np.vstack((np.array(trials).T, np.array(out))).T.tolist())

#%% SAVE DATA
#list of programming langauges
with open(PATH_TO_OUTFILE, 'w') as f:
    for element in seqs[0]:
        f.write('%s\n' % element)
