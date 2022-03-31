'''First pass analysing behavioral data'''

#%%
import os
import json
from pathlib import Path as pth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% general helper functions
def retrieve_data(path_to_file, strlist):
    """ Load in data saved from javascript online study
        Deals with missing '{}' and separates out saved keys
        into variables

    Parameters:
        path_to_file (str) : full path to file that contains data
        strlist (list) : list of strings with dict names to extract from files

    Returns:
        out (list) : list of dicts from file

    """

    out = []
    data_read = []
    with open(path_to_file, "r") as file:
        data_read = file.read().split('\n')
    data_read = [json.loads('{' + line + '}') for line in data_read]

    for i_dict, keyword in enumerate(strlist):
        out.append(data_read[i_dict][keyword])

    return out

# compute accuracy per block
def compute_bonus(responses, max_bonus):
    """ Computes total bonus earned based on series of responses labeled as
        correct/incorrect and given the maximum amount possible to achieve,
        assuming chance level at 50%
        and bonus to increase with each additional percent correct

    Parameters:
        responses (list) : list of correct or incorrect labels per trials
        max_bonus (int) : amount of bonus to earn given 100% accuracy

    Return:
        bonus_var (int) : amount of bonus earned given accuracy across responses

    """
    acc = np.nansum(responses)/len(responses)
    bonus_var = (((acc*100)-50)/50)*max_bonus

    return bonus_var

# find which trial was known based on generalised data.
def retrieve_uniqueness_point(seq_id):
    """ Finds index of first repeating sequence
        after all unique sequences have been shown at least once.
        (This is the latest trial at which a rule should be uniquely identified)

    Parameters:
        seq_id (list) : list of sequence ID for each trial in order of presentation
    Return:
        i_trial (int) : index (trial number)

    """
    all_trialid = np.unique(seq_id)
    seen_trialid = []
    i_trial = None
    for i_trial, trialid in enumerate(seq_id):
        if np.array_equal(np.unique(seen_trialid), all_trialid):
            break
        seen_trialid.append(trialid)

    return i_trial

#%%
## PRINT A BUNCH OF INFO TO GET A FEEL FOR HOW PARTICIPANT DID

def output_submission_details(df, fname):
    iloc = df_subject.index[df_subject['filename'] == fname].tolist()[0]
    #print out info from servers side
    print('Subject ID ' + df['filename'][iloc])
    print('Prolific ID ' + df['participant_id'][iloc])
    print('Group ' + df['condition'][iloc])
    print('Prolific time recorded: {0}'.format(np.round(df['time_taken'][iloc]/60)))
    print('Time to finish: %.2f minutes'% np.round(df['duration'][iloc], decimals=2))
    try:
        print('Time spend on instructions: %.2f'% np.round(df['instruction_time'][iloc], decimals=2))
    except TypeError:
        print('Time spend on instructions: unkown')

        #breaks taken
    for i_block, duration in enumerate(df['all_pause'][iloc]):
        print('spend {0} minutes on pause {1}'.format(np.round(duration, decimals=2), i_block))
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

        [t_data, e_data,parameters_data] = retrieve_data(f,['sdata','edata','parameters'])

        this_csv = all_csvs.loc[all_csvs['participant_id'] == e_data['expt_turker']]
        df_out = df_out.append(this_csv, ignore_index=True)
        all_files.append(file_name)
        all_rules.append(e_data['expt_group'])

        all_instruct_time.append((t_data['resp_timestamp'][1] - e_data['exp_starttime'])/1000/60)
        all_duration.append((e_data['exp_finishtime'] - e_data['exp_starttime'])/1000/60)
        pause_time = []
        for iblock, start_time in enumerate(e_data['block_starttime']):
            pause_time.append((e_data['block_finishtime'][iblock] - start_time)/1000/60)
        all_pause.append(pause_time)

        debrief_fields = [x for x in e_data.keys() if x.startswith('debrief')]
        tmp_debrief = []
        for field in debrief_fields:
            tmp_debrief.append(field + ' - ' +  e_data[field])
        all_debrief.append(tmp_debrief)

    df_out['filename'] = all_files
    df_out['condition'] = all_rules
    df_out['duration'] = all_duration
    df_out['instruction_time'] = all_instruct_time
    df_out['all_pause'] = all_pause
    df_out['debrief'] = all_debrief

    return df_out

## COLLECT BEHAVIORAL DATA
def fetch_data(path_to_data, list_of_filenames):
    df_out = pd.DataFrame(columns=['i_subject', 'group', 'block_num',
                                    'trial_num', 'rule', 'seqid',
                                    'correct', 'rt'])

    for isub, file_name in enumerate(list_of_filenames):
        #get participant data
        f = os.path.join(path_to_data, file_name)
        [trial_data, experiment_data, parameters_data] = retrieve_data(f,['sdata','edata','parameters'])

        #Check if participants saw 1-step test block
        if len(parameters_data['block']['trialorder'][-1]) < len(parameters_data['block']['trialorder'][-2]):
            #last block has fewer trials
            test_block = True

        for index in trial_data['expt_index']:
            if index == None:
                continue

            iblock = trial_data['expt_block'][index]
            itrial = trial_data['expt_trial'][index]

            row = pd.Series([
                file_name,
                experiment_data['expt_group'],
                iblock,
                trial_data['expt_trial'][index],
                parameters_data['ruleid'][iblock],
                #trial_data['seq'][index],
                parameters_data['block']['trialorder'][iblock][itrial-1],
                trial_data['resp_correct'][index],
                trial_data['resp_reactiontime'][index]
            ], index=df_out.columns)

            df_out = df_out.append(row, ignore_index=True)

    return df_out

#%%
path_data = "data/prolific/data/2step_v1"
path_demographic = "data/prolific/demographics"

df_subject = fetch_demographics(path_demographic, path_data)
df_data = fetch_data(path_data, df_subject['filename'].values)

#compute accuracy
df_acc = df_data[['i_subject', 'block_num', 'correct']].groupby(['i_subject', 'block_num']).agg(['sum', 'count'])
df_acc.columns = ['_'.join(column) for column in df_acc.columns]
df_acc['acc'] = df_acc['correct_sum']/df_acc['correct_count']
df_acc = df_acc['acc'].unstack()
filename = 'ZySjMhKVBPgI.txt'
for filename in  df_subject['filename'].values:
    start_date = df_subject[df_subject['filename'] == filename]['started_datetime']
    #if not start_date.str.startswith('2022-03-30').values[0]:
    #    continue

    print(filename)
    output_submission_details(df_subject, filename)
    print('PERFORMANCE: ')
    for iblock in range(len(df_acc.columns)):
        print(np.round(df_acc[iblock][filename], decimals=2))

#%% Visualize trial order, why did some people learn so well?
plt.figure(figsize=(30, 10))
for filename in  df_subject['filename'].values:
    dt_tmp = df_data[df_data['i_subject']==filename]
    acc = dt_tmp[dt_tmp['block_num']==2]['correct'].mean()
    if acc > 0.6:
        mycolor = 'r'
    else: mycolor = 'k'
    f = os.path.join(path_data, filename)
    [_, _, parameters_data] = retrieve_data(f, ['sdata','edata','parameters'])
    trials = parameters_data['block']['trialorder']
    trials_arr = list(range(len(trials)))
    plt.plot(trials_arr, trials,lw=2,color=mycolor)
    plt.title(filename)

#%% First pass, output info for each participant, this is to quickly check whether data quality is okay

for file_name in df_subject['filename'].values:
    path_file = os.path.join(path_data, file_name)

    #extract some variables from the data
    [s_data, e_data, p_data] = retrieve_data(path_file, ['sdata', 'edata', 'parameters'])
    trial_duration = p_data['timing']['seqduration']/1000

    single_data = df_data[df_data['i_subject'] == file_name]

    n_trials_presented = len(single_data)
    n_block = len(np.unique(single_data['block_num']))

    #Visualize performance

    fig = plt.figure(figsize=(30, 15))
    iblock = 0
    for iblock in np.unique(single_data['block_num']):
        if iblock == 11:
            break
        blocked_rt = single_data[single_data['block_num'] == iblock]['rt']
        idx_timeouts = [i for i, value in enumerate(blocked_rt) if value is None]
        blocked_correct = single_data[single_data['block_num'] == iblock]['correct']
        idx_incorrect = np.where(blocked_correct != 1)
        idx_unique = retrieve_uniqueness_point(single_data['seqid'])

        trials_arr = list(range(len(blocked_rt)))
        # set time-outs to 0 RT and subtract duration of trial from response counter
        blocked_rt = [0 if i is None else i-trial_duration for i in blocked_rt]
        y_min = np.min(blocked_rt)
        y_max = np.max(blocked_rt)

        idx_incorrect = [i for i in idx_incorrect[0] if i not in idx_timeouts]

        # plot RTs
        plt.subplot(n_block, 1, iblock+1)
        plt.plot(trials_arr, blocked_rt)
        plt.vlines(idx_incorrect, y_min, y_max, 'k')
        plt.vlines(idx_timeouts, y_min, y_max, 'r')
        plt.vlines(idx_unique, y_min, y_max, 'b')

        plt.ylim(y_min,y_max)
    fig.legend(['reaction time', 'incorrect trials', 'time_out', 'uniqueness point'], loc='lower center')
    fig.suptitle('subject {0} - group {1}'.format(set(single_data['i_subject']), set(single_data['group'])))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.savefig('results/prolific/2step_short/'+'summaryfig_' + file_name[:-4] + '.jpg')


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
