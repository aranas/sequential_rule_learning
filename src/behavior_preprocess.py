'''A collection of functions that extract data from the raw json files
  and save them to pandas arrays or operates on the resulting dataframe'''

import json
from pathlib import Path as pth
import os
import numpy as np
import pandas as pd

#%%
path_to_data = "data/prolific/data/2step_V2"
path_to_demographic = "data/prolific/demographics"
path_to_results = 'results/prolific/2step_V2/'

#  general helper functions
def data2dict(file_path):
    """ Load in data saved from javascript online study
        Deals with missing '{}' and outputs data in the form of dicts.

    Parameters:
        path_to_file (str) : full path to file that contains data

    Returns:
        out (list) : list of dicts from file

    """
    with open(file_path, "r") as file:
        data_read = file.read().split('\n')
    out = [json.loads('{' + line + '}') for line in data_read]

    return out

def dicts2df(list_of_dicts):

    all_dfs = []
    for mylist in list_of_dicts:
        df_tmp = pd.json_normalize(mylist, sep='-')
        all_dfs.append(df_tmp)
    df_data_1row = pd.concat(all_dfs, axis=1)

    #get column names that carry trial information (i.e. contain vector of max length)
    col_length = df_data_1row.apply(lambda col: len(np.array(col[0]).flatten()), axis=0)
    col_trl = col_length[col_length == max(col_length)].index.tolist()

    #unrol trial information
    df_data = df_data_1row[col_trl + ['edata-expt_turker']].set_index(['edata-expt_turker']).apply(pd.Series.explode).reset_index()
    #add non trl data
    df_repeat = df_data_1row.loc[:, ~df_data_1row.columns.isin(col_trl)]
    df_data.merge(df_repeat)

    return df_data

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
    iloc = df.index[df['filename'] == fname].tolist()[0]
    #print out info from servers side
    print('Subject ID ' + df['filename'][iloc])
    print('Prolific ID ' + df['participant_id'][iloc])
    print('Group ' + df['condition'][iloc])
    print('Prolific time recorded: {0}'.format(np.round(df['time_taken'][iloc]/60)))
    print('Time to finish: %.2f minutes'% np.round(df['duration'][iloc], decimals=2))
    try:
        print('Time spend on instructions: %.2f'% np.round(df['instruction_time'][iloc], decimals=2))
    except Exception:
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
    all_csvs = all_csvs[all_csvs['status'] != 'RETURNED']

    # check if anyone participated multiple times across experiments
    pp_ids = all_csvs['participant_id'].values.tolist()
    if not len(np.unique(pp_ids)) == len(pp_ids):
        print('### WARNING REPEATED PARTICIPATION ###')
        double_submission = {[x for x in pp_ids if pp_ids.count(x) > 1]}
        print(double_submission)

    return all_csvs

def pd2np(all_data, col_group):

    acc_data = all_data.set_index(col_group)['sdata-resp_correct']
    shape = tuple(map(len, acc_data.index.levels))
    arr = np.full(shape, np.nan)
    # fill it using Numpy's advanced indexing
    arr[tuple(acc_data.index.codes)] = acc_data.values.flat

    return arr
