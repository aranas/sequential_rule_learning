'''First pass analysing behavioral data'''

#%%
import os
import json
from pathlib import Path as pth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
def check_submission(df, filename):
        iloc = df_per_subject.index[df_per_subject['filename']== filename].tolist()[0]
        #print out info from servers side
        print('Subject ID ' + df['filename'][iloc])
        print('Prolific ID ' + df['participant_id'][iloc])
        print('Group ' + df['condition'][iloc])
        print('Time to finish: %.2f minutes'% np.round(df['duration'][iloc], decimals=2))
        print('Prolific time recorded: {0}'.format(np.round(df['time_taken'][iloc]/60)))
        try:
            print('Time spend on instructions: %.2f'% np.round(df['instruction_time'][iloc], decimals=2))
        except Exception:
            print('Time spend on instructions: unkown')

        #breaks taken
        for iblock, duration in enumerate(df['all_pause'][iloc]):
            print('spend {0} minutes on pause {1}'.format(np.round(duration,decimals=2),iblock))
        print('## Debrief ##')
        for field in df['debrief'][iloc]:
            print(field)

        #print out info from prolific side
        try:
            print('## Prolific ##')
            print('Country of Birth: ' + df['Country of Birth'][iloc])
            print('Age: {0}'.format(df['age'][iloc]))
        except Exception:
            print('no prolific data')

strlist = ['sdata','edata','parameters']
filename = os.path.join(path_to_data,'fF8yti4iKr78.txt')
def retrieve_data(filename, strlist):
    out  = []
    data = []
    with open(filename, "r") as f:
        data = f.read().split('\n')
    data = [json.loads('{' + line + '}') for line in data]

    for i, keyword in enumerate(strlist):
        out.append(data[i][keyword])

    return out

# compute accuracy per block
def compute_bonus(responses,max_bonus):
    bonus_var = 0
    for iblock, block_data in enumerate(responses):
        acc = np.nansum(block_data)/len(block_data)
        pay = (((acc*100)-50)/50)*(max_bonus/5)
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
    if 3 in input_array[0]:
        idx_rule = 'simple'
    else:
        idx_rule = 'complex'

    rule_names = np.array(['Input', 'X', 'state','None'])

    name_rule = [rule_names[i] for i in input_array]

    return idx_rule, name_rule

#%%
## COLLECT DEMOGRAPHIC DATA & CHECK FOR X-STUDY DUPLICATES
def fetch_demographics(path_to_demographic, path_to_data):

    files = pth(path_to_demographic).rglob("*.csv")
    all_csvs = [pd.read_csv(file) for file in files]
    all_csvs = pd.concat(all_csvs)

    # check if anyone participated multiple times across experiments
    if not len(np.unique(all_csvs['participant_id'].values)) ==len(all_csvs['participant_id'].values):
        print('### WARNING REPEATED PARTICIPATION ###')

    df_out = pd.DataFrame(columns = all_csvs.columns)
    all_files = []
    all_rules = []
    all_duration = []
    all_instruct_time = []
    all_pause = []
    all_debrief = []
    for file_name in os.listdir(path_to_data):
        f = os.path.join(path_to_data, file_name)
        if not os.path.isfile(f):
            continue

        [_, experiment_data,parameters_data] = retrieve_data(f,['sdata','edata','parameters'])

        this_csv = all_csvs.loc[all_csvs['participant_id'] == experiment_data['expt_turker']]
        df_out = df_out.append(this_csv, ignore_index=True)
        all_files.append(file_name)
        all_rules.append(experiment_data['expt_group'])

        try:
            all_instruct_time.append((trial_data['resp_timestamp'][1] - experiment_data['exp_starttime'])/1000/60)
            all_duration.append((experiment_data['exp_finishtime'] - experiment_data['exp_starttime'])/1000/60)
            pause_time = []
            for iblock, start_time in enumerate(experiment_data['block_starttime']):
                pause_time.append((experiment_data['block_finishtime'][iblock] - start_time)/1000/60)
            all_pause.append(pause_time)
        except Exception:
            all_instruct_time.append(None)
            all_duration.append(None)
            all_pause.append(None)

        debrief_fields = [x for x in experiment_data.keys() if x.startswith('debrief')]
        tmp_debrief = []
        for field in debrief_fields:
            tmp_debrief.append(field + ' - ' +  experiment_data[field])
        all_debrief.append(tmp_debrief)

    df_out['filename'] = all_files
    df_out['condition'] = all_rules
    df_out['duration'] = all_duration
    df_out['instruction_time'] = all_instruct_time
    df_out['all_pause'] = all_pause
    df_out['debrief'] = all_debrief

    return df_out

## COLLECT BEHAVIORAL DATA
def fetch_data(path_to_data,list_of_filenames):
    df_out = pd.DataFrame(columns=['i_subject','group',
                                     'block_num', 'trial_num', 'rule','seq','correct', 'rt'])

    for isub, file_name in enumerate(list_of_filenames):
        #get participant data
        f = os.path.join(path_to_data, file_name)
        [trial_data, experiment_data,parameters_data] = retrieve_data(f,['sdata','edata','parameters'])

        #Check if participants saw 1-step test block
        if len(parameters_data['block']['trialorder'][-1]) < len(parameters_data['block']['trialorder'][-2]):
            #last block has fewer trials
            test_block = True

        for itrial in trial_data['expt_index']:
            if itrial == None:
                continue

            iblock = trial_data['expt_block'][itrial]

            row = pd.Series([
                file_name,
                experiment_data['expt_group'],
                iblock,
                trial_data['expt_trial'][itrial],
                parameters_data['ruleid'][iblock],
                trial_data['seq'][itrial],
                trial_data['resp_correct'][itrial],
                trial_data['resp_reactiontime'][itrial]
            ], index=df_out.columns)

            df_out = df_out.append(row, ignore_index=True)

    return df_out

#%%
path_to_data = "data/prolific/data/2step_long"
path_to_demographic = "data/prolific/demographics"

df_per_subject = fetch_demographics(path_to_demographic, path_to_data)
df_per_subject[df_per_subject['condition']=='simple']
df_data = fetch_data(path_to_data, df_per_subject['filename'].values)

df_acc = df_data[['i_subject','block_num','correct']].groupby(['i_subject','block_num']).agg(['sum','count'])
df_acc.columns = ['_'.join(column) for column in df_acc.columns]
df_acc['acc'] = df_acc['correct_sum']/df_acc['correct_count']
df_acc = df_acc['acc'].unstack()

for filename in  df_per_subject['filename'].values:
    if not df_per_subject[df_per_subject['filename']==filename]['started_datetime'].str.startswith('2022-03-30').values[0]:
        continue
    print(filename)
    check_submission(df_per_subject, filename)
    print('PERFORMANCE: ')
    for iblock in range(len(df_acc.columns)):
        print(np.round(df_acc[iblock][filename],decimals=2))

#%% First pass, output info for each participant, this is to quickly check whether data quality is okay
#FIXME: double check that figures are being created correctly
#FIXME: double check that people (specifically those with smaller blocks) are seeing correct sequence - output combinations
# create some figures to show Chris
#FIXME: renalayse old data with new pipeline. Double check if trial order was highly biased in earlier pilots
for file_name in df_per_subject['filename'].values:
    [trial_data, experiment_data,parameters_data] = retrieve_data(os.path.join(path_to_data,file_name),['sdata','edata','parameters'])
    #Visualize performance
    trial_duration = parameters_data['timing']['seqduration']/1000
    num_timeouts = len([i for i in range(len(trial_data['resp_reactiontime'])) if trial_data['resp_reactiontime'][i] == None])

    single_data = df_data[df_data['i_subject']==file_name]
    n_block = len(np.unique(single_data['block_num']))
    fig = plt.figure(figsize=(15, 10))
    iblock = 0
    for iblock in np.unique(single_data['block_num']):
        blocked_rt = single_data[single_data['block_num']==iblock]['rt']
        blocked_correct = single_data[single_data['block_num']==iblock]['correct']
        trials_arr = list(range(len(blocked_rt)))
        ymin = 0#np.min(block_data)
        ymax = 10#np.max(block_data)
        # plot RTs
        plt.subplot(n_block, 1, iblock+1)
        plt.plot(trials_arr, blocked_rt.tolist())
        # mark incorrect trials
        idx_incorrect = np.where(blocked_correct != 1)
        plt.vlines(idx_incorrect, ymin, ymax, 'k')
        # mark first trial that can be inferred through completion based on the other trials
        #FIXMEplt.vlines(unique_ids[iblock], ymin, ymax, 'r')

        plt.ylim(ymin,ymax)
    fig.legend(['reaction time', 'incorrect trials'], loc='lower center')
    fig.suptitle('subject {0} - group {1}'.format(set(single_data['i_subject']), set(single_data['group'])))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    #plt.savefig('results/prolific/'+'RT_' + file_name[:-4] + '.jpg')


#bonus_sheet = pd.DataFrame({'id':all_names, 'pay':all_bonus})
#bonus_sheet.to_csv('bonus.csv')

#%%
# Visualize 1-step test block without feedback
for file_name in df_out['filename'].values:
    f = os.path.join(DATA_DIR, file_name)
    data = retrieve_data(f)

    trial_data  = data[0]['sdata']
    parameters_data = data[2]['parameters']
    n_testtrials = len(parameters_data['ruleid'][-1])
    print(n_testtrials)
    rule_hat    = np.array(trial_data['resp_category'][-n_testtrials*4:])
    rule        = np.array(trial_data['target_response'][-n_testtrials*4:])

    plt.figure()
    # state-input order per trial, rules are blocked: 0-0, 0-1, 1-0, 1-1
    tmp_count = 0
    for i in range(n_testtrials):
        tmp_count += 1
        tmp_idx = 4*i
        plt.subplot(n_testtrials,2,tmp_count)
        plt.imshow(rule[tmp_idx:tmp_idx+4].reshape(2,2))
        tmp_count += 1
        plt.subplot(n_testtrials,2,tmp_count)
        plt.imshow(rule_hat[tmp_idx:tmp_idx+4].reshape(2,2))

#%%

## Group Analysis pipeline
df_data = pd.DataFrame(columns=['i_subject','group',
                                 'block_num', 'rule','acc', 'mean_RT'])
for isub, file_name in enumerate(df_out['filename'].values):
    f = os.path.join(DATA_DIR, file_name)
    data = retrieve_data(f)

    nblock = parameters_data['nb_blocks']
    trial_data = data[0]['sdata']
    experiment_data = data[1]['edata']
    parameters_data = data[2]['parameters']

    ruleid, rulename = map_rule2ruleid(parameters_data['ruleid'])

    # preprocess
    n_testtrials = len(parameters_data['ruleid'][-1])
    nblock = parameters_data['nb_blocks']-n_testblock
    if n_testblock is 1 :
        all_correct = trial_data['resp_correct'][1:-n_testtrials*4]
    else:
        all_correct = trial_data['resp_correct'][1:]

    blocked_correct = np.array_split(all_correct, nblock)
    blocked_rt      = np.array_split(trial_data['resp_reactiontime'][1:], nblock)

    acc_blocked = []
    rt_blocked = []
    count_rep_blocked = []
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

    num_timeouts = len([i for i in range(len(trial_data['resp_reactiontime'])) if trial_data['resp_reactiontime'][i] == None])
    if num_timeouts > 4:
        #print('exclude subject {0}from group {1}'.format(file_name,experiment_data['expt_group']))
        print('timeouts {0}'.format(str(num_timeouts)))
        #print(acc_blocked)
        #continue

    for iblock, block_data in enumerate(blocked_correct):
        block_name ='_'.join(['block',str(iblock)])
        row = pd.Series([
            file_name,
            experiment_data['expt_group'],
            block_name,
            parameters_data['ruleid'][iblock],
            acc_blocked[iblock],
            rt_blocked[iblock],
        ], index=df_data.columns)
        df_data = df_data.append(row, ignore_index=True)

mean_acc = df_data.groupby(['block_num','group']).agg(['mean',np.nanstd,'count'])
df_data.replace('block_0','1',inplace=True)
df_data.replace('block_1','2',inplace=True)
df_data.replace('block_2','3',inplace=True)
df_data.replace('block_3','4',inplace=True)

#%%
# Draw a scatter plot
import seaborn as sns

plt.figure()
g = sns.catplot(x='block_num',y='acc', hue='group',data=df_data, kind="box")
g.map_dataframe(sns.stripplot, x="block_num", y="acc",
                hue="group", alpha=0.6, dodge=True)

plt.figure()
g = sns.lineplot(x='block_num',y='acc', hue='group',data=df_data, err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('main effect accuracy learning')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc.jpg',bbox_inches="tight")

plt.figure()
g=sns.lineplot(x='block_num',y='acc_corrected', hue='group',data=df_plot, err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('main effect accuracy - corrected for reps')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc_noreps.jpg',bbox_inches="tight")


plt.figure()
g=sns.lineplot(x='block_num',y='mean_RT', hue='group',data=df_plot, err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('main effect RTs - corrected for reps')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_RT_noreps.jpg',bbox_inches="tight")


plt.figure()
g=sns.lineplot(x='block_num',y='acc', style='last_block', hue='group',data=df_plot, err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('detailed effects acc')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc_detailed.jpg',bbox_inches="tight")

plt.figure()
g=sns.lineplot(x='block_num',y='acc', style='curriculum', data=df_plot[df_plot['group']=='control'], err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('Control group only - ACC')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc_detailed2.jpg',bbox_inches="tight")


plt.figure()
g=sns.lineplot(x='block_num',y='acc_corrected', style='last_block', hue='group',data=df_plot, err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('detailed effects acc - corrected')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'main_acc_detailed_corrected.jpg',bbox_inches="tight")

# Are some rule easier to learn than others?
plt.figure()
sns.stripplot(x='first_block',y='acc',data=df_plot[df_plot['block_num']=='block_0'],jitter=True)
plt.title('rule difficulty in 1st block')
plt.savefig('results/prolific/group/'+'rule_difficulty.jpg',bbox_inches="tight")

plt.figure()
sns.stripplot(x='first_block',y='acc_corrected',data=df_plot[df_plot['block_num']=='block_0'],jitter=True)
plt.title('rule difficulty in 1st block - corrected for reps')
plt.savefig('results/prolific/group/'+'rule_difficulty_corrected.jpg',bbox_inches="tight")

plt.figure()
g=sns.lineplot(x='block_num',y='acc_rest',  hue='lucky_guess',data=df_plot, err_style='bars', err_kws={'capsize':6}, marker='o')
plt.legend(['1st incorrect','1st correct'],bbox_to_anchor=(1.02, 0.55),loc='upper left',borderaxespad=0)
plt.title('main effect accuracy - divided by success on first trial')
g.set_xticklabels(['1','2','3','4'])
plt.savefig('results/prolific/group/'+'split_firsttrial.jpg',bbox_inches="tight")

#%%
from scipy.stats import mannwhitneyu

import itertools
# create an empty dictionary
test_results = {}

group1 = df_data.where((df_data.group == 'complex') & (df_data.block_num == 'block_5')).dropna()
group2 = df_data.where((df_data.group== 'simple') & (df_data.block_num == 'block_5')).dropna()
# add the output to the dictionary
test_results['main'] = mannwhitneyu(group1['acc_corrected'],group2['acc_corrected'])


#%%
