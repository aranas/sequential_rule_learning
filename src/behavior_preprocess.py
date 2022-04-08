'''A collection of functions that exctract data from the raw json files
  and save them to pandas arrays'''

import json
from pathlib import Path as pth
import os
import numpy as np
import pandas as pd

#  general helper functions
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

    return acc,bonus_var

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

        #all_instruct_time.append((t_data['resp_timestamp'][1] - e_data['exp_starttime'])/1000/60)
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
    #df_out['instruction_time'] = all_instruct_time
    df_out['all_pause'] = all_pause
    df_out['debrief'] = all_debrief

    return df_out

## COLLECT BEHAVIORAL DATA
def fetch_data(path_to_data, list_of_filenames):
    df_out = pd.DataFrame(columns=['i_subject', 'group', 'block_num',
                                    'trial_num', 'rule', 'seqid', 'response',
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
                trial_data['resp_category'][index],
                trial_data['resp_correct'][index],
                trial_data['resp_reactiontime'][index]
            ], index=df_out.columns)

            df_out = df_out.append(row, ignore_index=True)

    return df_out
