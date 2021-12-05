# %%
# First attempt to train Elman RNN on sequential rule task,
# where the task is to predict the value of a binary output state, given an initial state and sequentially appearing input-operator pairs
import sys
sys.path.append('/Users/sophiearana/Documents/Work/Projects/SeqLearning/code')
import numpy as np
import random
import seqlearn_magicspell as magic

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import itertools

import matplotlib.pyplot as plt
from tabulate import tabulate

#%%
class SeqData(Dataset):
    def __init__(self, data, labels, seq_len):
        'Initialization'
        self.data = data
        self.seq_len = seq_len
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sequence = self.data[index,:].astype(np.float32)
        out_state = self.labels[index].astype(np.float32)

    #FIXME: turn into one-hot encoding within this class

        return sequence, out_state

def convert_seq2onehot(seq,n_ops):

    data = []
    for trial in seq:
        trial_data = []
        for t in trial:
            init = torch.tensor([0,0])
            if not np.isnan(t[0]):
                init = torch.nn.functional.one_hot(torch.tensor(t[0]), num_classes=2)
            input = torch.tensor([0,0])
            if not np.isnan(t[1]):
                input = torch.nn.functional.one_hot(torch.tensor(t[1]), num_classes=2)
            op = torch.tensor([0]*n_ops)
            if not np.isnan(t[2]):
                #op = torch.tensor([0])
                op = torch.nn.functional.one_hot(torch.tensor(t[2]), num_classes=n_ops)
            inputvec = torch.cat((init,input,op),0)
            trial_data.append(inputvec)
        data.append(torch.stack(trial_data))
    data = torch.stack(data,dim=0) #combine into tensor of shape n_trial X n_time X n_inputvector
    data = data.numpy()

    return data

def convert_onehot2seq(seq):
    new_seq = []
    for trial in seq:
        tmp = []
        for i,step in enumerate(trial):
            if i==0:
                init = int(np.where(step[0:2])[0])
                tmp.append((init,np.nan,np.nan))
                continue
            else:
                cue = int(np.where(step[2:4])[0])
                rule = int(np.where(step[4:])[0])
                tmp.append((np.nan,cue,rule))
        new_seq.append(tmp)
    return new_seq

def generate_data(example_obs,len_seq):

    example_rules = np.stack([magic.dRules[i]['rule'] for i in example_obs])
    n_ops = len(example_obs)

    ### Create input stimuli & convert to 1-hot ###
    seq = magic.generate_trial(example_obs,len_seq, replacement=True)
    out = magic.transform(seq,example_obs,magic.dRules)
    inputvec = convert_seq2onehot(seq,n_ops)

    ### Create train-test split ###
    seqdata = SeqData(inputvec, out, len_seq)
    train_set_size = int(len(seqdata) * 0.8)
    test_set_size = len(seqdata) - train_set_size

    trainset, testset = random_split(seqdata, [train_set_size, test_set_size])
    trainseq = trainset[:][0]
    trainout = trainset[:][1]
    trainseq = convert_onehot2seq(trainseq)
    recovery_prob = magic.model_recovery(trainseq,trainout,example_obs,[2,4],[0,5,10],1000)

    return trainset, testset

class OneStepRNN(nn.Module):

    def __init__(self, D_in, D_out, recurrent_size, hidden_size):
        super(OneStepRNN, self).__init__()
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(D_in + self.recurrent_size, self.recurrent_size)
        self.input2fc1 = nn.Linear(D_in + self.recurrent_size, self.hidden_size)  # size input, size output
        self.fc1tooutput = nn.Linear(self.hidden_size, 2)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), dim=0)
        self.hidden = nn.functional.relu(self.input2hidden(combined))
        self.fc1_activations = nn.functional.relu(self.input2fc1(combined))
        self.output = self.fc1tooutput(self.fc1_activations)
        return self.output.view(-1,2), self.hidden

    def get_activations(self, x, hidden):
        self.forward(x, hidden)  # update the activations with the particular input
        return self.hidden, self.fc1_activations, self.output

    def get_noise(self):
        return self.hidden_noise

    def initHidden(self):
        return torch.zeros(1, self.recurrent_size)[0]

def train(sequence,label,model,optimizer,criterion):
    optimizer.zero_grad()
    #Read each cue in and keep hidden state for next cue
    hidden = model.initHidden()
    for i in range(len(sequence[0])):
        output, hidden = model.forward(sequence[0][i], hidden)
    #Compare final output to target
    #loss = criterion(torch.unsqueeze(output,dim=0), label.long())
    loss = criterion(output,label.long())
    #Back-propagate
    loss.backward()
    optimizer.step()

    return output, loss.item()

#%%
len_seq = 2
example_obs = [0,1]
def run(len_seq,example_obs):
    # define input data parameters
    #len_seq = 1
    #example_obs = [0,1]
    # define model & training parameters
    len_inputvec = len(example_obs)+4
    batchSize = 1
    recurrentSize = 4
    hiddenSize = 6
    learningRate = 0.001
    epochs = 3000
    rulenames = [magic.dRules[i]['name'] for i in example_obs]

    dattrain, dattest = generate_data(example_obs,len_seq)
    trainloader = DataLoader(dattrain, batch_size=batchSize, shuffle=True)
    testloader = DataLoader(dattest, batch_size=batchSize, shuffle=True)

    model = OneStepRNN(len_inputvec,1,recurrentSize, hiddenSize)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

# viz
#for x,y in trainloader:
#    break
#writer = SummaryWriter('runs/seq_learntest')
#hState = torch.zeros(1,recurrentSize)[0]
#writer.add_graph(model,(x[0][0],hState))
#writer.close()

    model.train()
    lossHistory = []
    for epoch in range(epochs):
        lossTotal = 0
        for x,y in trainloader:
            output, loss = train(x,y,model,optimizer,criterion)
            lossTotal +=loss
        lossHistory.append(lossTotal)

    model.eval()
    correct = 0
    for x,y in testloader:
        hidden = torch.zeros(1, recurrentSize)[0]
        for step in x[0]:
            hidden,h_activation,y_hat = model.get_activations(step,hidden)
        correct += int(y.detach().numpy()==np.where(y_hat==y_hat.max()))
        #print(y, y_hat)
    print('accuracy: %f ' % ((correct/len(testloader))))

    plt.plot(lossHistory)
    plt.title('Loss - rules {0} and {1} steps, acc on test = {2}'.format(rulenames,len_seq,(correct/len(testloader))))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show()
    plt.savefig('../figures/loss_SRN_{0}_{1}'.format(example_obs,len_seq))


if __name__ == '__main__':
    print('train elman RNN on magic spell task \n')
    print('the training data is generated using rules "reverse" & "force A", sequences were of length 2')

    len_seq = 2
    rule = [0,1]

    seq = magic.generate_trial(rule,len_seq, replacement=True)
# give summary of parameters
print("There are {0} unique sequences (operator-input combinations) of length {2}, \
given {1} true underlying rules \
and sampling with replacement.".format(len(seq),len(rule),len_seq))
obs_key = list(itertools.permutations(range(6),len(rule)))
print("There are {0} possible assignments of {1} possilbe rules onto the {2} operator symbols".format(len(obs_key),6,len(example_obs)))
#for rule in obs_key:
run(len_seq,rule)
