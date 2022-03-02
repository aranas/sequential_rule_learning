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
    print('Prolific time recorded: {0}'.format(np.round(this_csv['time_taken'].values[0]/60)))
    #instruct_time = experimentdata['instruct_finishtime'] - experimentdata['exp_starttime']/1000/60
    try:
        instruct_time = (trial_data['resp_timestamp'][1] - experiment_data['exp_starttime'])/1000/60
    except TypeError:
        instruct_time = (trial_data['resp_timestamp'][2] - experiment_data['exp_starttime'])/1000/60
    print('Time spend on instructions: %.2f'% np.round(instruct_time, decimals=2))
    #breaks taken
    for iblock, start_time in enumerate(experiment_data['block_starttime']):
        pause_time = (experiment_data['block_finishtime'][iblock] - start_time)/1000/60
        print('spend {0} minutes on pause {1}'.format(np.round(pause_time,decimals=2),iblock))
    print('## Debrief ##')
    print('used a tool: ' + experimentdata['debrief_tools'])
    print(experimentdata['debrief_feedback'])

    #print out info from prolific side
    try:
        print('## Prolific ##')
        print('Country of Birth: ' + this_csv['Country of Birth'].values[0])
        print('Age: {0}'.format(this_csv['age'].values[0]))
    except Exception:
        print('no prolific data')

    for iblock, block_data in enumerate(blocked_correct):
        acc = np.nansum(block_data)/len(block_data)
        print('{0} accuracy in block {1}'.format(acc, iblock))

def retrieve_data(filename):
    out = []
    with open(filename, "r") as f:
        out = f.read().split('\n')
    out = [json.loads('{' + line + '}') for line in out]

    return out

# compute accuracy per block
def compute_bonus(responses,max_bonus):
    bonus_var = 0
    for iblock, block_data in enumerate(responses):
        acc = np.nansum(block_data)/len(block_data)
        pay = (((acc*100)-50)/50)*(max_bonus/4)
        bonus_var = bonus_var+pay
    return bonus_var

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
# read in prolific data
files = pth("data/prolific/demographics").rglob("*.csv")
all_csvs = [pd.read_csv(file) for file in files]
all_csvs = pd.concat(all_csvs)

DATA_DIR = "data/prolific/data"
all_dates = []
all_files = []
for file_name in os.listdir(DATA_DIR):
    f = os.path.join(DATA_DIR, file_name)
    if not os.path.isfile(f):
        continue
    data = retrieve_data(f)
    experiment_data = data[1]['edata']
    this_csv = all_csvs.loc[all_csvs['participant_id'] == experiment_data['expt_turker']]
    all_dates.append(this_csv['started_datetime'].values[0])
    all_files.append(file_name)
df_files = pd.DataFrame({'filename':all_files, 'date&time':all_dates})
df_files = df_files.sort_values('date&time')

groups = []
all_names = []
all_bonus = []
for file_name in df_files['filename'].values:
    f = os.path.join(DATA_DIR, file_name)
    data = retrieve_data(f)

    trial_data = data[0]['sdata']
    experiment_data = data[1]['edata']
    parameters_data = data[2]['parameters']
    # preprocess
    nblock = parameters_data['nb_blocks']
    all_correct = trial_data['resp_correct'][1:]
    # for now ignore timeouts
    all_correct = pd.DataFrame(np.array(all_correct)).fillna(0).to_numpy().flatten()
    blocked_correct = np.array_split(all_correct, nblock)

    reaction_times = np.array_split(trial_data['resp_reactiontime'][1:], nblock)
    reaction_times = pd.DataFrame(np.array(reaction_times)).fillna(0).to_numpy()

    check_submission(file_name, experiment_data, parameters_data, blocked_correct)

    unique_ids = retrieve_uniqueness_point(parameters_data['block']['trialorder'])

    #Visualize performance
    trial_duration = parameters_data['timing']['seqduration']/1000
    num_timeouts = len([i for i in range(len(trial_data['resp_reactiontime'])) if trial_data['resp_reactiontime'][i] == None])
    print('{0} trials timed out'.format(num_timeouts))

    trials_arr = list(range(len(reaction_times[0])))
    fig = plt.figure(figsize=(15, 10))
    for iblock, block_data in enumerate(reaction_times):
        block_data = block_data-(trial_duration)
        ymin = 0#np.min(block_data)
        ymax = 5#np.max(block_data)
        # plot RTs
        plt.subplot(nblock, 1, iblock+1)
        plt.plot(trials_arr, block_data)
        # mark incorrect trials
        idx_incorrect = np.where(blocked_correct[iblock] != 1)
        plt.vlines(idx_incorrect, ymin, ymax, 'k')
        # mark first trial that can be inferred through completion based on the other trials
        plt.vlines(unique_ids[iblock], ymin, ymax, 'r')

        plt.ylim(ymin,ymax)
    fig.legend(['reaction time', 'incorrect trials', 'uniqueness point'], loc='lower center')
    fig.suptitle('subject {0} - group {2} mean RT was {1}'.format(experiment_data['expt_turker'], np.round(np.mean(reaction_times), decimals=1), experiment_data['expt_group']))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    groups.append(parameters_data['ruleid'])
    all_names.append(experiment_data['expt_turker'])
    all_bonus.append(compute_bonus(blocked_correct, 1))


bonus_sheet = pd.DataFrame({'id':all_names, 'pay':all_bonus})
bonus_sheet.to_csv('bonus.csv')
