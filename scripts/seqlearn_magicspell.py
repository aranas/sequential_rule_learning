#%%
import itertools
import copy
import collections
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

# %%

#define all contingency tables
#rows refer to latent state, columns refer to input cue (A or B)
dRules = [
    {'name':'none', 'rule':np.array([[0, 0], [1, 1]]), 'ruletype':'none'},
    {'name':'reverse', 'rule':np.array([[1, 1], [0, 0]]), 'ruletype':'state'},
    {'name':'forceA', 'rule':np.array([[1, 0], [1, 0]]), 'ruletype':'input'},
    {'name':'forceB', 'rule':np.array([[0, 1], [0, 1]]), 'ruletype':'input'},
    {'name':'crossA', 'rule':np.array([[1, 0], [0, 1]]), 'ruletype':'X'},
    {'name':'crossB', 'rule':np.array([[0, 1], [1, 0]]), 'ruletype':'X'},
    {'name':'except1', 'rule':np.array([[1, 0], [0, 0]]), 'ruletype':'X_assym'},
    {'name':'except2', 'rule':np.array([[0, 1], [0, 0]]), 'ruletype':'X_assym'},
    {'name':'except3', 'rule':np.array([[0, 0], [1, 0]]), 'ruletype':'X_assym'},
    {'name':'except4', 'rule':np.array([[0, 0], [0, 1]]), 'ruletype':'X_assym'},
    {'name':'exceptR1', 'rule':np.array([[0, 1], [1, 1]]), 'ruletype':'X_assym'},
    {'name':'exceptR2', 'rule':np.array([[1, 0], [1, 1]]), 'ruletype':'X_assym'},
    {'name':'exceptR3', 'rule':np.array([[1, 1], [0, 1]]), 'ruletype':'X_assym'},
    {'name':'exceptR4', 'rule':np.array([[1, 1], [1, 0]]), 'ruletype':'X_assym'}
]

# Create an example assignment using the first 3 rules in the ruledict

def generate_trial(operators, input_ids, len_seq, replacement=False, for_js_script=False):
    ''' This function defines all possible permutations of init state & sequence
    of binary input cues and operators.
    Output is an array of shape n X len_seq+1.
    Each row is one of n unique ordered permutations.
    First column indicates the initital state (binary) at t=0.
    Every following column contains a tuple, where the first position indicates
    the binary input cue and the second position indicates the operator identity
    The first element'''
    seq = []
    if replacement:
        combi_inputcue = list(itertools.product(input_ids, repeat=len_seq))
        combi_operators = list(itertools.product(operators, repeat=len_seq))
    else:
        if len_seq == 2: # if seq of 2 sample binary cue evenly
            combi_inputcue = list(itertools.permutations(input_ids, len_seq))
        else:
            combi_inputcue = list(itertools.product(input_ids, repeat=len_seq))
        combi_operators = list(itertools.permutations(operators, len_seq))

    for init in range(2):
        for cue in combi_inputcue:
            for op in combi_operators:
                if not for_js_script:
                    seq.append([(init, np.nan, np.nan),
                                *zip([np.nan]*len(cue),
                                     cue, tuple(op))]) #group per time point t
                else:
                    seq.append([[init], *zip(op, cue)])

    return seq

def transform(trials, all_rules, **kwargs):
    true_rules = kwargs.get('true_rules', None)
    new_rules = kwargs.get('new_rules', None)
    if new_rules is not None:
        all_rules[true_rules[0]], all_rules[true_rules[1]] = all_rules[new_rules[0]], all_rules[new_rules[1]]

    outputs = []
    for _, trial in enumerate(trials):
        #print('trial # %i' % itrial)
        #print(trial)
        for i, x in enumerate(trial):
            if i == 0:
                state = x[0]
                continue
            cue = x[1]
            while cue > 1:
                cue -= 2
            id_operator = x[0]

            rule = all_rules[id_operator]['rule']
            #print('current state %i and current cue %i' %(state, cue))
            #print(rule)
            state = rule[state, cue]
            #print('new state %i' % state)

        outputs.append(state)
    return np.array(outputs)

def model_recovery(seq, out, true_rules, all_reps, noiselevels, shuffles, verbose=False):
    rng = default_rng(5)
    # for a given set of sequences (data), return which rule assignments
    # out of all possible combinations of all_rules
    # is ambiguous with the true_rules
    all_obs = np.stack([d['rule'] for d in dRules[0:6]], axis=2).transpose(2, 0, 1)

    all_out = []
    obs_key = list(itertools.permutations(range(len(all_obs)), len(true_rules)))
    for combination in obs_key:
        #replace rules:
        all_out.append(transform(seq, copy.copy(dRules),
                                    true_rules=true_rules, new_rules=combination))

    RDM = np.empty([len(all_reps), len(noiselevels), shuffles, len(all_out)])
    for r, nrepeats in enumerate(all_reps):
        ntrials = len(seq)*nrepeats
        out_alt = np.tile(all_out, nrepeats)
        out_true = np.tile(out, nrepeats)
        out_alt = np.vstack((out_alt, out_true))
        #noisel = 0
        for i, noisel in enumerate(noiselevels):
            nerrors = round(((len(seq)*nrepeats)/100)*noisel)
            # simulate data repeatedly
            for sh in range(shuffles):
                # for any given ground truth model (row), add noise
                if noisel == 0:
                    noisel = 1
                noisy_data = copy.copy(out_true)
                id_noisy = rng.choice(ntrials, nerrors, replace=False)
                noisy_data[id_noisy] = 1-noisy_data[id_noisy]
                # compute likelihood for all models
                tmpsum = np.zeros(out_alt.shape)
                tmpsum[np.where(noisy_data == out_alt)] = 1-noisel*0.01
                tmpsum[np.where(noisy_data != out_alt)] = noisel*0.01
                likelihood = (np.prod(tmpsum, axis=1)).reshape(len(tmpsum), 1)
                    # decide which model has more evidence
                if any(likelihood == 0):
                    RDM[r, i, sh, :] = np.nan
                else:
                    LR = np.log(likelihood/np.transpose(likelihood))
                    RDM[r, i, sh, :] = LR[-1][0:-1]

    if verbose:
        ratio = np.nansum((RDM > 0).astype(int), 2)/shuffles
        # Find out which rules are confounded
        tmprdm = ratio[0, 0, :]
        numUniq = 182 - sum(tmprdm)
        id_confound = np.where(tmprdm == 0)[0]
        print('{0} ambiguous rules given data of rule {1}'.format(len(id_confound)-1,
                                                                    (dRules[true_rules[0]]['name'],dRules[true_rules[1]]['name'])))
        confounded = []
        for t in id_confound:
            r1 = obs_key[t]
            if r1 == true_rules:
                continue
            if len(r1) == 1:
                print(dRules[t]['name'])
            else:
                print((dRules[r1[0]]['name'], dRules[r1[1]]['name']))
            confounded.append(r1)

    return RDM

# %%
    #len_seq = 2
    #operators = [0,1,2]
    #model = np.stack([dRules[i]['rule'] for i in operators])
    #trials = generate_trial(operators,len_seq, replacement=False)
    #out = transform(trials,dRules)

if __name__ == '__main__':
    # Task parameters:
    len_seq = 2    #length of operator sequence shown on each trial
    n_ops = 2       #number of operator symbols seen
    n_rules = 6    #number possible rule taken into account

        # Ground truth assignment
    ideal_obs = np.stack([d['rule'] for d in dRules[0:n_ops]], axis=2).transpose(2, 0, 1)
    ideal_key = list(range(n_ops))

        # All possible rule assignment strategies
    all_obs = np.stack([d['rule'] for d in dRules[0:n_rules]], axis=2).transpose(2 , 0, 1)
    obs_key = list(itertools.permutations(range(n_rules), n_ops))

    observers = []
    for combination in obs_key:
        new_observer = all_obs[list(combination)]
        observers.append(new_observer)

        # Generate all possible sequences + output according to ideal observer
    seq = generate_trial(ideal_key, [0, 1], len_seq, replacement=True)
    out = transform(seq, dRules)

        # give summary of parameters
    print("There are {0} unique sequences (operator-input combinations) of length {2}, \
    given {1} true underlying rules \
    and sampling with replacement.".format(len(seq), n_ops, len_seq))

    print("There are {0} possible assignments of {1} possilbe rules onto the {2} operator symbols".format(len(obs_key), n_rules, n_ops))

        ### Model recovery ###

    shuffles = 1000
    all_reps = list([4, 8, 10])
    noiselevels = list([0, 2, 5, 10])
    RDM = []
    for obs in obs_key:
    #for data given each possible ground-truth model get likelihood ratio for all others
        seq = generate_trial(obs, [0, 1], len_seq, replacement=True)
        y = transform(seq, dRules)
        RDM.append(model_recovery(seq, y, obs, all_reps, noiselevels, shuffles, verbose=False))
    RDM = np.stack(RDM, axis=4)
    # What is the probability of finding correct model given noisy data? Across multiple simulations
    ratio = np.nansum((RDM > 0).astype(int), 2)/RDM.shape[2]

    # Find out which rules are confounded
    tmprdm = ratio[0, 0, :, :]
    for i, row in enumerate(tmprdm):
        tmprdm[i, i] = np.nan
    numUniq = 182 - np.nansum(tmprdm, axis=0)
    nconfound = collections.Counter(numUniq)
    id_confound = np.where(tmprdm == 0)
    seen = set()
    confoundpairs = [t for t in zip(id_confound[0], id_confound[1]) if tuple(sorted(t)) not in seen and not seen.add(tuple(sorted(t)))]
    print('{0} rule assignments are non-ambiguous.'.format(nconfound[1]))
    print('confounded rule assignments:')
    for t in confoundpairs:
     r1 = obs_key[t[0]]
     r2 = obs_key[t[1]]
     print('{0} and {1}'.format((dRules[r1[0]]['name'], dRules[r1[1]]['name']), (dRules[r2[0]]['name'], dRules[r2[1]]['name'])))


    ratio[ratio == 0] = 0.5

    mean_RDM = np.nanmean(RDM, axis=2).astype(int)
    #Visualize
    fig, axs = plt.subplots(RDM.shape[0], len(noiselevels), figsize=(20, 20))
    for r, reps in enumerate(all_reps):
        for i, noisel in enumerate(noiselevels):
            im = axs[r, i].imshow(ratio[r, i, :, :], cmap='magma', interpolation=None)
            axs[r, i].set_title('{0}% noise and {1} trials'.format(noisel, reps*len(seq)))
            fig.colorbar(im, ax=axs[r, i])
    plt.show()
