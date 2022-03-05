'''First pass analysing behavioral data'''

#%%
import os
import json
from pathlib import Path as pth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%

def check_submission(filename, experimentdata, parametersdata, blocked_correct):
        #print out info from servers side
        print('Subject ID ' + filename)
        print('Prolific ID ' + experimentdata['expt_turker'])
        print('Group ' + experimentdata['expt_group'])
        ruleid, rulename = map_rule2ruleid(parameters_data['ruleid'])
        print('Rule ID: {0}'.format(rulename))
        try:
            finish_time = (experimentdata['exp_finishtime'] - experimentdata['exp_starttime'])/1000/60
            print('Time to finish: %.2f minutes'% np.round(finish_time, decimals=2))
        except KeyError:
            print('did not finish')
        print('Prolific time recorded: {0}'.format(np.round(this_csv['time_taken'].values[0]/60)))
        #instruct_time = experimentdata['instruct_finishtime'] - experimentdata['exp_starttime']/1000/60
        try:
            instruct_time = (trial_data['resp_timestamp'][1] - experiment_data['exp_starttime'])/1000/60
            print('Time spend on instructions: %.2f'% np.round(instruct_time, decimals=2))
        except Exception:
            print('not timestamp close enough to estimate instruction time')
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

def map_rule2ruleid(input_array):
    rule_names = np.array(['Input', 'X', 'state'])
    rules = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 2, 1, 0], [2, 1, 2, 0],[0, 2, 0, 1], [2, 0, 2, 1]])

    idx_rule = np.where((rules == input_array).all(axis=1))[0][0]
    name_rule = [rule_names[i] for i in input_array]

    return idx_rule, name_rule

#%%
# read in prolific data
files = pth("data/prolific/demographics").rglob("*.csv")
all_csvs = [pd.read_csv(file) for file in files]
all_csvs = pd.concat(all_csvs)

if not len(np.unique(all_csvs['participant_id'].values)) ==len(all_csvs['participant_id'].values):
    print('### WARNING DOUBLE PARTICIPANT ###')

# check if anyone participated multiple times across experiments
DATA_DIR = "data/prolific/data"
all_dates = []
all_files = []
all_rules = []
for file_name in os.listdir(DATA_DIR):
    f = os.path.join(DATA_DIR, file_name)
    if not os.path.isfile(f):
        continue
    data = retrieve_data(f)
    experiment_data = data[1]['edata']
    parameters_data = data[2]['parameters']
    this_csv = all_csvs.loc[all_csvs['participant_id'] == experiment_data['expt_turker']]
    ruleid, rulename = map_rule2ruleid(parameters_data['ruleid'])
    all_dates.append(this_csv['started_datetime'].values[0])
    all_files.append(file_name)
    all_rules.append(ruleid)

df_files = pd.DataFrame({'filename':all_files, 'group': all_rules, 'date&time':all_dates})
df_files = df_files.sort_values('date&time')

#%%
groups = []
all_names = []
all_bonus = []
condition = df_files['date&time'].str.startswith('2022-03-04')
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
    reaction_times = pd.DataFrame(reaction_times).fillna(0).to_numpy()

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
    plt.savefig('results/prolific/'+'RT_' + file_name[:-4] + '.jpg')

    groups.append(parameters_data['ruleid'])
    all_names.append(experiment_data['expt_turker'])
    all_bonus.append(compute_bonus(blocked_correct, 1))

groups
bonus_sheet = pd.DataFrame({'id':all_names, 'pay':all_bonus})
bonus_sheet.to_csv('bonus.csv')

#%%

## Analysis pipeline
df_data = pd.DataFrame(columns=['i_subject','group', 'curriculum', 'block_num', 'rule','acc','num_reps','mean_RT','lucky_guess','second_trial'])
for isub, file_name in enumerate(df_files['filename'].values):
    f = os.path.join(DATA_DIR, file_name)
    data = retrieve_data(f)

    nblock = parameters_data['nb_blocks']
    trial_data = data[0]['sdata']
    experiment_data = data[1]['edata']
    parameters_data = data[2]['parameters']

    ruleid, rulename = map_rule2ruleid(parameters_data['ruleid'])

    all_correct = trial_data['resp_correct'][1:]
    blocked_correct = np.array_split(all_correct, nblock)
    blocked_rt      = np.array_split(trial_data['resp_reactiontime'][1:], nblock)


    acc_blocked = []
    rt_blocked = []
    count_rep_blocked = []
    lucky_guess = []
    second_trial = []
    for iblock, block_data in enumerate(blocked_correct):
        #num of repeating adjacent trials
        trialorder = parameters_data['block']['trialorder'][iblock]
        count_rep = 0
        for idx, seq_num in enumerate(trialorder[:-1]):
            if seq_num == trialorder[idx+1]:
                count_rep += 1
        count_rep_blocked.append(count_rep)
        #FIXME: maybe rather want to treat timeouts as incorrect?
        acc_blocked.append(pd.Series(block_data).mean())
        rt_blocked.append(pd.Series(blocked_rt[iblock]).mean())
        lucky_guess.append(block_data[0])
        second_trial.append(block_data[1])

    num_timeouts = len([i for i in range(len(trial_data['resp_reactiontime'])) if trial_data['resp_reactiontime'][i] == None])
    if num_timeouts > 4:
        print('exclude subject {0}from group {1}'.format(file_name,experiment_data['expt_group']))
        print('timeouts {0}'.format(str(num_timeouts)))
        print(acc_blocked)
        continue

    if np.all(0.7 > np.array(acc_blocked)):
        print('exclude subject {0}from group {1}'.format(file_name,experiment_data['expt_group']))
        print(acc_blocked)
        continue
    for iblock, block_data in enumerate(blocked_correct):
        block_name ='_'.join(['block',str(iblock)])
        row = pd.Series([
            file_name,
            experiment_data['expt_group'],
            '-'.join(rulename),
            block_name,
            parameters_data['ruleid'][iblock],
            acc_blocked[iblock],
            count_rep_blocked[iblock],
            rt_blocked[iblock],
            lucky_guess[iblock],
            second_trial[iblock]
        ], index=df_data.columns)
        df_data = df_data.append(row, ignore_index=True)

mean_acc = df_data.groupby(['block_num','group','curriculum']).agg(['mean',np.nanstd,'count'])
df_data['second_trial'] = pd.to_numeric(df_data['second_trial'])
two_shot = df_data.groupby(['block_num','group']).agg('sum')


#%%
# Draw a scatter plot
import seaborn as sns
#colors = df_data['rule'].map(dict(rule_1='green', rule_2='red',rule_3='blue', rule_4='yellow'))
df_plot = df_data.copy()
df_plot['last_block'] = df_plot.curriculum.apply(lambda x: x.split('-')[-1])
df_plot['first_block'] = df_plot.curriculum.apply(lambda x: x.split('-')[1])
df_plot.replace('X-X-X-X','X',inplace=True)
df_plot.replace('Input-Input-Input-Input','Input',inplace=True)
df_plot.replace('state-X-state-Input','state_Input',inplace=True)
df_plot.replace('X-State-X-Input','X_Input',inplace=True)
df_plot.replace('state-Input-state-X','state_X',inplace=True)
df_plot.replace('Input-state-Input-X','Input_X',inplace=True)
df_plot['rule'].replace('0','Input',inplace=True)
df_plot['rule'].replace('1','X',inplace=True)
df_plot['rule'].replace('2','State',inplace=True)
df_plot['acc_corrected'] = df_plot['acc'] - 0.025*df_plot['num_reps']

#plt.hist(df_data[df_data['group']=='generalise']['acc'])
#plt.hist(df_data[df_data['group']=='control']['acc'])


plt.figure()
g = sns.lineplot(x='block_num',y='acc', hue='group',data=df_plot, err_style='bars', marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('main effect accuracy learning')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc.jpg',bbox_inches="tight")

plt.figure()
g=sns.lineplot(x='block_num',y='acc_corrected', hue='group',data=df_plot, err_style='bars', marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('main effect accuracy - corrected for reps')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc_noreps.jpg',bbox_inches="tight")


plt.figure()
g=sns.lineplot(x='block_num',y='mean_RT', hue='group',data=df_plot, err_style='bars', marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('main effect RTs - corrected for reps')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_RT_noreps.jpg',bbox_inches="tight")


plt.figure()
g=sns.lineplot(x='block_num',y='acc', style='last_block', hue='group',data=df_plot, err_style='bars', marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('detailed effects acc')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc_detailed.jpg',bbox_inches="tight")

plt.figure()
g=sns.lineplot(x='block_num',y='acc', style='rule', hue='group',data=df_plot, err_style='bars', marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('detailed effects acc')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc_detailed2.jpg',bbox_inches="tight")


plt.figure()
g=sns.lineplot(x='block_num',y='acc_corrected', style='last_block', hue='group',data=df_plot, err_style='bars', marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('detailed effects acc - corrected')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc_detailed_corrected.jpg',bbox_inches="tight")

# Are some rule easier to learn than others?
plt.figure()
sns.stripplot(x='first_block',y='acc',data=df_plot[df_plot['block_num']=='block_0'],jitter=True)
plt.title('rule difficulty in 1st block')
plt.figure()
sns.stripplot(x='first_block',y='acc_corrected',data=df_plot[df_plot['block_num']=='block_0'],jitter=True)
plt.title('rule difficulty in 1st block - corrected for reps')
plt.savefig('results/prolific/group/'+'rule_difficulty_corrected.jpg',bbox_inches="tight")

plt.figure()
g=sns.lineplot(x='block_num',y='acc',  style='lucky_guess',data=df_plot, err_style='bars', marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('main effect accuracy - divided by success on first trial')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'split_firsttrial.jpg',bbox_inches="tight")

#%%
from scipy.stats import mannwhitneyu

from scipy.stats import f_oneway

import itertools
# create an empty dictionary
test_results = {}

group1 = df_data.where((df_data.group == 'generalise') & (df_data.block_num == 'block_3')).dropna()
group2 = df_data.where((df_data.group== 'control') & (df_data.block_num == 'block_3')).dropna()
# add the output to the dictionary
test_results['main'] = mannwhitneyu(group1['acc'],group2['acc'])
f_oneway(group1['acc'].values,group2['acc'].values)

group1 = df_data.where((df_data.lucky_guess == 0) & (df_data.block_num != 'block_2')).dropna()
group2 = df_data.where((df_data.lucky_guess == 1) & (df_data.block_num != 'block_2')).dropna()
mannwhitneyu(group1['acc'],group2['acc'])
f_oneway(group1['acc'].values,group2['acc'].values)

curricula_order = list(set(df_data['curriculum']))
# loop over column_list and execute code explained above
for block in set(df_data['block_num']):
    group = []
    for curr in curricula_order:
        group.append(df_data.where((df_data.curriculum == curr)&(df_data.block_num == block)).dropna()['acc'].values)

f_oneway(group1['acc'].values,group2['acc'].values)
f_oneway(group[0],group[1],group[2],group[3],group[4],group[5])

#%%
