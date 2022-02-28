'''First pass analysing behavioral data'''

#%%
import os
import json
from pathlib import Path as pth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%

rule_names = np.array(['force', 'cross', 'reverse'])

def check_submission(filename, experimentdata, parametersdata, blocked_correct):
    #print out info from servers side
    print('Subject ID ' + filename)
    print('Prolific ID ' + experimentdata['expt_turker'])
    print('Group ' + experimentdata['expt_group'])
    print('Rule ID: {0}'.format(rule_names[np.unique(parametersdata['ruleid'])]))
    finish_time = (experimentdata['exp_finishtime'] - experimentdata['exp_starttime'])/1000/60
    print('Time to finish: %.2f minutes'% np.round(finish_time, decimals=2))
    #instruct_time = experimentdata['instruct_finishtime'] - experimentdata['exp_starttime']/1000/60
    instruct_time = (trial_data['resp_timestamp'][1] - experimentdata['exp_starttime'])/1000/60
    print('Time spend on instructions: %.2f'% np.round(instruct_time, decimals=2))

    #print out info from prolific side
    try:
        print('## Prolific ##')
        files = pth("data/prolific/demographics").rglob("*.csv")
        all_csvs = [pd.read_csv(file) for file in files]
        all_csvs = pd.concat(all_csvs)
        all_csvs.columns.values.tolist()
        this_csv = all_csvs.loc[all_csvs['participant_id'] == experiment_data['expt_turker']]
        print('Prolific time recorded: {0}'.format(np.round(this_csv['time_taken'].values[0]/60)))
        print('Country of Birth: ' + this_csv['Country of Birth'].values[0])
        print('Age: {0}'.format(this_csv['age'].values[0]))
    except Exception:
        print('no prolific data')
    bonus_var = 0
    for iblock, block_data in enumerate(blocked_correct):
        acc = np.nansum(block_data)/len(block_data)
        pay = (((acc*100)-50)/50)*(3/4)
        bonus_var = bonus_var+pay
        print('{0} accuracy in block {1}, bonus is {2}'.format(acc, iblock, pay))
    print('Total bonus: {0}'.format(bonus_var))

def retrieve_data(filename):
    out = []
    with open(filename, "r") as f:
        out = f.read().split('\n')
    out = [json.loads('{' + line + '}') for line in out]

    return out

# compute accuracy per block
def acc_per_block(responses_blocked):
    acc_arr = []
    for block_data in responses_blocked:
        acc = np.sum(block_data)/len(block_data)
        acc_arr.append(acc)
    return acc_arr

# find which trial was known based on generalised data.
def retrieve_uniqueness_point(blocked_trialorder):
    all_trialid = np.unique(blocked_trialorder[0])
    uniqueness_point = []
    for trialorder in blocked_trialorder:
        seen_trialid = []
        for itrial, trialid in enumerate(trialorder):
            if np.array_equal(np.unique(seen_trialid), all_trialid):
                break
            seen_trialid.append(trialid)
        uniqueness_point.append(itrial)
    return uniqueness_point


#%%
DATA_DIR = "data/prolific/data"

for file_name in os.listdir(DATA_DIR):
    f = os.path.join(DATA_DIR, file_name)
    data = retrieve_data(f)

    trial_data = data[0]['sdata']
    experiment_data = data[1]['edata']
    parameters_data = data[2]['parameters']
    #preprocess
    nblock = parameters_data['nb_blocks']
    all_correct = trial_data['resp_correct'][1:]
    # for now ignore timeouts
    all_correct = pd.DataFrame(np.array(all_correct)).fillna(0).to_numpy().flatten()
    blocked_correct = np.array_split(all_correct, nblock)

    check_submission(file_name, experiment_data, parameters_data, blocked_correct)

    unique_ids = retrieve_uniqueness_point(parameters_data['block']['trialorder'])

    #Visualize performance
    cut_off = 4
    reaction_times = np.array_split(trial_data['resp_reactiontime'][1:], nblock)
    trial_duration = parameters_data['timing']['seqduration']/1000

    num_above_threshold = len(np.where(trial_data['resp_reactiontime'][1:] > np.round((trial_duration + cut_off)))[0])
    print('{0} trials would be rejected at cut_off {1}'.format(num_above_threshold, cut_off))

    trials_arr = list(range(len(reaction_times[0])))
    fig = plt.figure(figsize=(15, 10))
    for iblock, block_data in enumerate(reaction_times):
        block_data = block_data-(trial_duration)
        ymin = 0#np.min(block_data)
        ymax = 10#np.max(block_data)
        # plot RTs
        plt.subplot(nblock, 1, iblock+1)
        plt.plot(trials_arr, block_data)
        # mark incorrect trials
        idx_incorrect = np.where(blocked_correct[iblock] != 1)
        plt.vlines(idx_incorrect, ymin, ymax, 'k')
        # mark first trial that can be inferred through completion based on the other trials
        plt.vlines(unique_ids[iblock], ymin, ymax, 'r')

        plt.ylim(ymin,ymax)
    fig.legend(['reaction time', 'uniqueness point', 'incorrect trials'], loc='lower center')
    fig.suptitle('subject {0} - mean RT was {1}'.format(file_name,np.round(np.mean(reaction_times), decimals=1)))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
