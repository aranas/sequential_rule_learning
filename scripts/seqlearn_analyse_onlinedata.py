'''First pass analysing behavioral data'''

#%%
import json
import numpy as np

# %%

data = []
with open("/Library/WebServer/Documents/tasks/magic_spell/data/data/ftUnKjSskK66.txt", "r") as f:
    data = f.read().split('\n')
data = [json.loads('{' + line + '}') for line in data]

trial_data = data[0]['sdata']
trial_data.keys()
experiment_data = data[1]['edata']
experiment_data.keys()

parameters_data = data[2]['parameters']
parameters_data.keys()

# %%
# print basic information (prolific ID, experimental group, time it took to finish)
#print(experiment_data['expt_turker'])
#print(experiment_data['expt_group'])
#print('%i minutes' % np.round((experiment_data['exp_finishtime'] - experiment_data['exp_starttime'])/1000/60,decimals=0))

#%%
# compute accuracy per block

trial_correct = trial_data['resp_correct'][1:] # first response is in instructions
blocked_correct = np.array_split(trial_correct, 4)

for iblock, block_data in enumerate(blocked_correct):
    acc = np.sum(block_data)/len(block_data)
    print('{0} accuracy in block {1}'.format(acc, iblock))

#%%
# compute if performance on 1st possible generalisable item

# find which trial was known based on generalised data.
last_block = parameters_data['block']['trialorder'][-1]

all_trialid = np.unique(last_block)
seen_trialid = []
for itrial, trialid in enumerate(last_block):
    seen_trialid.append(trialid)
    if np.array_equal(np.unique(seen_trialid), all_trialid):
        break
# correct or not?
print(blocked_correct[-1][itrial])
