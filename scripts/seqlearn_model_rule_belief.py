'''Model which belief over the rules people have at any given time'''

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import src.behavior_preprocess as prep

#%%

def transform_1step(trial, all_rules):
    """ Compute outcome according to each rule in all_rules
        for a 1-step state-ingr sequence

    Parameters:
        trial (list) : list of steps in sequence [state, ingr]
        all_rules (dict) : each entry in all_rules corresponds to one possible rule
                        A rule is specificed under key "rule", where value is
                        a 2-by-2 matrix
    Return:
        outputs (arr) : Outcomes for each rule

    """
    outputs = []
    state = trial[0]
    ingr = trial[1]
    for operator in all_rules:
        rule_mat = operator['rule']
        outstate = rule_mat[state, ingr]
        outputs.append(outstate)
    return np.array(outputs)

#%%

path_data = "data/prolific/data/v3"
path_demographic = "data/prolific/demographics"
path_results = 'results/prolific/ruleBelief/'

df_subject = prep.fetch_demographics(path_demographic, path_data)
df_data = prep.fetch_data(path_data, df_subject['filename'].values)

f = os.path.join(path_data, df_subject['filename'].values[0])
[_, _, parameters_data] = prep.retrieve_data(f, ['sdata', 'edata', 'parameters'])

sequences = parameters_data['seq']
rules = parameters_data['rules']

#%%
dRules = [
    {'name':'force', 'rule':np.array([[0, 1], [0, 1]]), 'ruletype':'input'},
    {'name':'cross', 'rule':np.array([[0, 1], [1, 0]]), 'ruletype':'X'},
    {'name':'reverse', 'rule':np.array([[1, 1], [0, 0]]), 'ruletype':'state'},
    {'name':'except1', 'rule':np.array([[0, 1], [1, 1]]), 'ruletype':'X_assym'},
    {'name':'except2', 'rule':np.array([[1, 0], [1, 1]]), 'ruletype':'X_assym'},
    {'name':'except3', 'rule':np.array([[1, 1], [0, 1]]), 'ruletype':'X_assym'},
    {'name':'except4', 'rule':np.array([[1, 1], [1, 0]]), 'ruletype':'X_assym'}
]

WIN_WIDTH = 10

df_general = df_data[df_data['group'] == 'generalise']
for subject in set(df_general['i_subject']):

    fig, axs = plt.subplots(1, 4, figsize=(40, 10))
    for i_block, block_num in enumerate(set(df_general['block_num'])):
        tmp_data = df_general[(df_general['i_subject'] == subject) &
                              (df_general['block_num'] == block_num)]

        n_trials = len(tmp_data)

        block_rule = dRules[np.unique(tmp_data['rule'])[0]]['name']
        resp_cat = tmp_data['response']

        for rule in dRules:

            all_out_states = []
            for index, row in tmp_data.iterrows():
                seq = sequences[row['seqid']]
                out_state_per_rule = transform_1step(seq, [rule])
                all_out_states.append(out_state_per_rule[0])
            correct_per_rule = np.equal(all_out_states, resp_cat).astype('int').tolist()

            fit_acc = []
            for i_slice in range(n_trials-WIN_WIDTH):
                acc = sum(correct_per_rule[i_slice:i_slice+WIN_WIDTH])/WIN_WIDTH
                fit_acc.append(acc)

            axs[i_block].plot(range(len(fit_acc)), fit_acc, label=rule['name'])
            axs[i_block].title.set_text('rule {0}'.format(block_rule))
        if i_block == 3:
            axs[i_block].legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
        fig.suptitle('subject {0} - block {1}'.format(subject, i_block))
