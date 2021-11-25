#%%
import numpy as np
import random
import itertools
import copy
from numpy.random import default_rng
import collections
import matplotlib.pyplot as plt

# %%

#define all contingency tables
#rows refer to latent state, columns refer to input cue (A or B)
dRules = [
    {'name':'op_none', 'rule':np.array([[0,0],[1,1]]), 'ruletype':'state_only'},
    {'name':'op_reverse', 'rule':np.array([[1,1],[0,0]]), 'ruletype':'state_only'},
    {'name':'op_forceA', 'rule':np.array([[1,0],[1,0]]),'ruletype':'input_only'},
    {'name':'op_forceB', 'rule':np.array([[0,1],[0,1]]),'ruletype':'input_only'},
    {'name':'op_crossA', 'rule':np.array([[1,0],[0,1]]),'ruletype':'interaction'},
    {'name':'op_crossB', 'rule':np.array([[0,1],[1,0]]),'ruletype':'interaction'},
    {'name':'op_except1', 'rule':np.array([[1,0],[0,0]]),'ruletype':'interaction'},
    {'name':'op_except2', 'rule':np.array([[0,1],[0,0]]),'ruletype':'interaction'},
    {'name':'op_except3', 'rule':np.array([[0,0],[1,0]]),'ruletype':'interaction'},
    {'name':'op_except4', 'rule':np.array([[0,0],[0,1]]),'ruletype':'interaction'},
    {'name':'op_exceptR1', 'rule':np.array([[0,1],[1,1]]),'ruletype':'interaction'},
    {'name':'op_exceptR2', 'rule':np.array([[1,0],[1,1]]),'ruletype':'interaction'},
    {'name':'op_exceptR3', 'rule':np.array([[1,1],[0,1]]),'ruletype':'interaction'},
    {'name':'op_exceptR4', 'rule':np.array([[1,1],[1,0]]),'ruletype':'interaction'}
]

# Create an example assignment using the first 3 rules in the ruledict

def generate_trial(operators,len_seq,replacement=False):
    # This function defines all possible permutations of init state & sequence of binary input cues and operators.
    # Output is an array of shape n X len_seq+1.
    # Each row is one of n unique ordered permutations.
    # First column indicates the initital state (binary) at t=0.
    # Every following column contains a tuple, where the first position indicates the binary input cue and the second position indicates the operator identity
    # The first element
    seq = []
    if replacement:
        combi_inputcue = list(itertools.product([0,1],repeat=len_seq))
        combi_operators = list(itertools.product(range(len(operators)),repeat=len_seq))
    else:
        if len_seq == 2: # if seq of 2 sample binary cue evenly
            combi_inputcue = list(itertools.permutations([0,1],len_seq))
        else:
            combi_inputcue = list(itertools.product([0,1],repeat=len_seq))
        combi_operators = list(itertools.permutations(range(len(operators)),len_seq))

    for init in range(2):
        for cue in combi_inputcue:
            for op in combi_operators:
                seq.append([(init,np.nan,np.nan),*zip([np.nan]*len(cue),cue,tuple(op))]) #group per time point t
                #seq.append([init] + list(cue) + list(op))

    return seq

def transform(trials,rule_sel,allRules):
    outputs = []
    for itrial,trial in enumerate(trials):
        #print('trial # %i' % itrial)
        #print(trial)
        for i,x in enumerate(trial):
            if i is 0:
                state = x[0]
                continue
            cue = x[1]
            id_operator = x[2]

            rule = allRules[rule_sel[id_operator]]['rule']
            #print('current state %i and current cue %i' %(state,cue))
            #print(rule)
            state = rule[state,cue]
            #print('new state %i' % state)

        outputs.append(state)
    return np.array(outputs)

def model_recovery(seq,out,true_rules,all_reps,noiselevels,shuffles):
    rng = default_rng(5)
    # for a given set of sequences (data), return which rule assignments
    # out of all possible combinations of all_rules
    # is ambiguous with the true_rules
    all_obs = np.stack([d['rule'] for d in dRules[0:6]],axis=2).transpose(2,0,1)

    all_out = []
    obs_key = list(itertools.permutations(range(len(all_obs)), len(true_rules)))
    for combination in obs_key:
        all_out.append(transform(seq,combination,dRules))

    RDM = np.empty([len(all_reps),len(noiselevels),shuffles,len(all_out)])
    for r,nrepeats in enumerate(all_reps):
        ntrials = len(seq)*nrepeats
        out_alt = np.tile(all_out,nrepeats)
        out_true = np.tile(out,nrepeats)
        out_alt = np.vstack((out_alt,out_true))
        #noisel = 0
        for i,noisel in enumerate(noiselevels):
            nerrors = round(((len(seq)*nrepeats)/100)*noisel)
            # simulate data repeatedly
            for sh in range(shuffles):
                # for any given ground truth model (row), add noise
                if noisel ==0: noisel=1
                id_noisy = rng.choice(ntrials,nerrors,replace=False)
                out_true[id_noisy] = 1-out_true[id_noisy]
                # compute likelihood for all models
                tmpsum = np.zeros(out_alt.shape)
                tmpsum[np.where(out_true == out_alt)] = 1-noisel*0.01
                tmpsum[np.where(out_true != out_alt)] = noisel*0.01
                likelihood = (np.prod(tmpsum,axis=1)).reshape(len(tmpsum),1)
                    # decide which model has more evidence
                if any(likelihood==0):
                    RDM[r,i,sh,:] = np.nan
                else:
                    LR = np.log(likelihood/np.transpose(likelihood))
                    RDM[r,i,sh,:] = LR[-1][0:-1]

    ratio = np.nansum((RDM>0).astype(int),2)/shuffles
    # Find out which rules are confounded
    tmprdm = ratio[0,0,:]
    numUniq = 182 - sum(tmprdm)
    id_confound = np.where(tmprdm==0)[0]
    print('{0} ambiguous rules given these data'.format(len(id_confound)))
    confounded=[]
    for t in id_confound:
        r1 = obs_key[t]
        if len(r1)==1:
            print(dRules[t]['name'])
        else:
            print((dRules[r1[0]]['name'],dRules[r1[1]]['name']))
        confounded.append(r1)

    return RDM

# %%
    #len_seq = 2
    #operators = [0,1,2]
    #model = np.stack([dRules[i]['rule'] for i in operators])
    #trials = generate_trial(operators,len_seq, replacement=False)
    #out = transform(trials,operators,dRules)

if __name__ == '__main__':
    # Task parameters:
    len_seq = 2    #length of operator sequence shown on each trial
    n_ops = 2       #number of operator symbols seen
    n_rules = 6    #number possible rule taken into account

    # Ground truth assignment
    ideal_obs = np.stack([d['rule'] for d in dRules[0:n_ops]],axis=2).transpose(2,0,1)
    ideal_key = list(range(n_ops))

    # All possible rule assignment strategies
    all_obs = np.stack([d['rule'] for d in dRules[0:n_rules]],axis=2).transpose(2,0,1)
    obs_key = list(itertools.permutations(range(n_rules), n_ops))

    observers=[]
    for combination in obs_key:
        new_observer = all_obs[list(combination)]
        observers.append(new_observer)

    # Generate all possible sequences + output according to ideal observer
    seq = generate_trial(ideal_key,len_seq, replacement=True)
    out = transform(seq,list(range(n_ops)),dRules)

    # give summary of parameters
    print("There are {0} unique sequences (operator-input combinations) of length {2}, \
    given {1} true underlying rules \
    and sampling with replacement.".format(len(seq),n_ops,len_seq))

    print("There are {0} possible assignments of {1} possilbe rules onto the {2} operator symbols".format(len(obs_key),n_rules,n_ops))

    ### Model recovery ###
    ## FIXME: after code cleanup model recovery gives different results than before
    shuffles = 1000
    all_reps = list([4,8,10])
    noiselevels = list([0,2,5,10])

    RDM = []
    for obs in obs_key: #for data given each possible ground-truth model get likelihood ratio for all others
        y = transform(seq,obs,dRules)
        RDM.append(model_recovery(seq,y,obs,all_reps,noiselevels,shuffles))
    RDM = np.stack(RDM,axis=4)
    # What is the probability of finding correct model given noisy data? Across multiple simulations
    ratio = np.nansum((RDM>0).astype(int),2)/RDM.shape[2]
    ratio[ratio==0] = 0.5

    mean_RDM = np.nanmean(RDM,axis=2).astype(int)
    #Visualize
    fig, axs = plt.subplots(RDM.shape[0], len(noiselevels),figsize=(20,20))
    for r,reps in enumerate(all_reps):
        for i,noisel in enumerate(noiselevels):
            im = axs[r,i].imshow(mean_RDM[r,i,:,:],cmap='magma',interpolation=None)
            axs[r,i].set_title('{0}% noise and {1} trials'.format(noisel,reps*len(seq)))
            fig.colorbar(im,ax=axs[r,i])
    plt.show()
