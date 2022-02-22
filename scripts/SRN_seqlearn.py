# %%
# First attempt to train Elman RNN on sequential rule task,
# where the task is to predict the value of a binary output state, given an initial state and sequentially appearing input-operator pairs
import sys
sys.path.append('/Users/sophiearana/Documents/Work/Projects/SeqLearning/code')
import numpy as np
import pandas as pd
import random
import seqlearn_magicspell as magic
import time
import pickle
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

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

def convert_seq2onehot(seq,n_ops,n_inputs):

    data = []
    for trial in seq:
        trial_data = []
        for t in trial:
            init = torch.tensor([0,0])
            if not np.isnan(t[0]):
                init = torch.nn.functional.one_hot(torch.tensor(t[0]), num_classes=2)
            input = torch.tensor([0]*n_inputs)
            if not np.isnan(t[1]):
                input = torch.nn.functional.one_hot(torch.tensor(t[1]), num_classes=n_inputs)
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

def convert_onehot2seq(seq,n_ops,n_inputs):
    new_seq = []
    for trial in seq:
        tmp = []
        for i,step in enumerate(trial):
            if i==0:
                init = int(np.where(step[0:2])[0])
                tmp.append((init,np.nan,np.nan))
                continue
            else:
                cue = int(np.where(step[2:2+n_inputs])[0])
                rule = int(np.where(step[2+n_inputs:])[0])
                tmp.append((np.nan,cue,rule))
        new_seq.append(tmp)
    return new_seq

def generate_data(example_obs,input_ids,len_seq,train_split):

    example_rules = np.stack([magic.dRules[i]['rule'] for i in example_obs])
    n_ops = len(example_obs)

    ### Create input stimuli & convert to 1-hot ###
    seq = magic.generate_trial(example_obs,input_ids,len_seq, replacement=True)
    out = magic.transform(seq,magic.dRules)
    inputvec = convert_seq2onehot(seq,6,4)

    ### Create train-test split ###
    seqdata = SeqData(inputvec, out, len_seq)
    train_set_size = int(len(seqdata) * train_split)
    test_set_size = len(seqdata) - train_set_size

    trainset, testset = random_split(seqdata, [train_set_size, test_set_size])
    #trainseq = trainset[:][0]
    #trainout = trainset[:][1]
    #trainseq = convert_onehot2seq(trainseq,6,4)
    #recovery_prob = magic.model_recovery(trainseq,trainout,example_obs,[4,8],[0,5,10],1000)

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
    #clipping gradients
    #torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10, norm_type=2.0)

    return output, loss.item()

#%%

def run(len_seq,example_obs):
    # define input data parameters
    #len_seq = 1
    #example_obs = [0,1]
    # define model & training parameters
    len_inputvec = 2+4+6
    batchSize = 1
    recurrentSize = 4
    hiddenSize = 6
    learningRate = 0.001
    epochs = 3000
    rulenames = [magic.dRules[i]['name'] for i in example_obs]

    #hardcoded: assume
    dattrain, dattest = generate_data(example_obs,[0,1],len_seq,0.8)
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

def train_generalise(len_seq,ops1,ops2,ops3,nseed):
        # define model & training parameters
        len_inputvec = 2+4+6 #binary input state, 4 input cues, 6 possible operators
        batchSize = 1
        recurrentSize = 4
        hiddenSize = 6
        learningRate = 0.001
        epochs = 500
        rulenames = [magic.dRules[i]['name'] for i in ops1]

        model = OneStepRNN(len_inputvec,1,recurrentSize, hiddenSize)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

        #print('create training data')
        dattrain1, tmp = generate_data(ops1,[0,1],len_seq,1) # rule 1, inputs A&B
        dattrain2, tmp = generate_data(ops2,[2,3],len_seq,1) # rule 2, inputs C&D
        dattest, tmp = generate_data(ops1,[2,3],len_seq,1) # rule 1, inputs C&D
        dattest2, tmp = generate_data(ops3,[2,3],len_seq,1) # rule 3, inputs C&D

        #dattrain = torch.utils.data.ConcatDataset([dattrain1,dattrain2])

        trainloader1 = DataLoader(dattrain1, batch_size=batchSize, shuffle=True)
        trainloader2 = DataLoader(dattrain2, batch_size=batchSize, shuffle=True)
        testloader = DataLoader(dattest, batch_size=batchSize, shuffle=True)
        testloader2 = DataLoader(dattest2, batch_size=batchSize, shuffle=True)
    # viz
    #for x,y in trainloader:
    #    break
    #writer = SummaryWriter('runs/seq_learntest')
    #hState = torch.zeros(1,recurrentSize)[0]
    #writer.add_graph(model,(x[0][0],hState))
    #writer.close()
        random.seed(nseed)
        model.train()
        # train on rule 1 given inputs A & B
        #print('train on rule {0}, given inputs A&B'.format( [magic.dRules[i]['name'] for i in ops1]))
        lossHistory = []
        for epoch in range(epochs):
            lossTotal = 0
            for x,y in trainloader1:
                output, loss = train(x,y,model,optimizer,criterion)
                lossTotal +=loss
            lossHistory.append(lossTotal)
        # train on rule 2 given inputs C & D
        #print('train on rule {0}, given inputs C&D'.format([magic.dRules[i]['name'] for i in ops2]))
        random.seed(nseed)
        lossHistory2 = []
        for epoch in range(epochs):
            lossTotal = 0
            for x,y in trainloader2:
                output, loss = train(x,y,model,optimizer,criterion)
                lossTotal +=loss
            lossHistory2.append(lossTotal)

        # train on rule 1 given inputs C & D
        #print('train on rule {0}, given inputs C&D'.format([magic.dRules[i]['name'] for i in ops1]))
        model_copy = copy.copy(model)
        random.seed(nseed)
        lossHistory3 = []
        for epoch in range(epochs):
            lossTotal = 0
            for x,y in testloader:
                output, loss = train(x,y,model_copy,optimizer,criterion)
                lossTotal +=loss
            lossHistory3.append(lossTotal)

        # train on rule 3 given inputs C & D
        lossHistory4 = []
        if ops3 is not None:
            #print('train on rule {0}, given inputs C&D'.format([magic.dRules[i]['name'] for i in ops3]))

            model_copy = copy.copy(model)
            random.seed(nseed)
            for epoch in range(epochs):
                lossTotal = 0
                for x,y in testloader2:
                    output, loss = train(x,y,model_copy,optimizer,criterion)
                    lossTotal +=loss
                lossHistory4.append(lossTotal)
        # train on rule 1 given inputs A&B again
        lossHistory5 = []
        for epoch in range(epochs):
            lossTotal = 0
            for x,y in trainloader1:
                output, loss = train(x,y,model_copy,optimizer,criterion)
                lossTotal +=loss
            lossHistory5.append(lossTotal)

        #model.eval()
        #correct = 0
        #for x,y in testloader:
        #    hidden = torch.zeros(1, recurrentSize)[0]
        #    for step in x[0]:
        #        hidden,h_activation,y_hat = model.get_activations(step,hidden)
        #    correct += int(y.detach().numpy()==np.where(y_hat==y_hat.max()))
            #print(y, y_hat)
        #print('accuracy: %f ' % ((correct/len(testloader))))

        #plt.plot(lossHistory+lossHistory2)
        #plt.title('Loss - rules {0} on new inputs, acc on test = {1}'.format(rulenames,(correct/len(testloader))))
        #plt.xlabel('Epoch')
        #plt.ylabel('Loss')
        #plt.savefig('../figures/lossgeneral_SRN_{0}_{1}'.format(ops1,ops2))
        return np.stack((lossHistory,lossHistory2,lossHistory3,lossHistory4,lossHistory5),axis=1)

def run_generalise_1step(ops1,ops2,ops3):
    # Train on single rules, learn a two rules blockwise sequentially each with different inputs
    # Test on generalisation of each rule to the new inputs
    # train on R1 with inputs A&B, then train on R2 with inputs C&D, test on R1 C&D or R2 A&B

    len_seq = 1
    rulenames = [magic.dRules[i]['name'] for i in [ops1,ops2,ops3]]
    legendtxt = ['Task 1 - %s'%rulenames[0],'Task 2 - %s'%rulenames[1],'Task 3 - generalise %s'%rulenames[0],'Task 3 - control %s'%rulenames[2], 'Task 3 - control interference']

    losses = []
    for i in range(50):
        print("random init # {0}".format(i))
        losses.append(train_generalise(len_seq,[ops1],[ops2],[ops3],i))
    losses = (np.stack(losses,axis=2))
    with open('../results/lossgeneral_{0}_{1}_{2}.txt'.format(ops1,ops2,ops3),"wb") as fp:
        pickle.dump(losses, fp)
    print("file saved under ../results/lossgeneral_{0}_{1}_{2}.txt".format(ops1,ops2,ops3))

    nepoch,nblock,nshuf = losses.shape

    plt.figure()
    for i in range(nblock):
        df = pd.DataFrame(np.transpose(losses[:,i,:],[1,0])).melt()
        sns.lineplot(x="variable", y="value", data=df)
    plt.legend(labels=legendtxt)
    plt.ylim([0,8])
    plt.xlabel('training epochs')
    plt.ylabel('training loss')
    plt.savefig('../figures/lossgeneral_{0}_{1}_{2}.jpg'.format(ops1,ops2,ops3))
    print("figure saved under ../figures/lossgeneral_{0}_{1}_{2}.jpg".format(ops1,ops2,ops3))

def run_generalise_2step(pair1,pair2,pair3,pair4):
    # Train on single rules, learn a two rules blockwise sequentially each with different inputs
    # Test on generalisation of each rule to the new inputs
    # train on R1 with inputs A&B, then train on R2 with inputs C&D, test on R1 C&D or R2 A&B

    len_seq = 2
    rulenames = [magic.dRules[i]['name'] for i in [pair1[0],pair1[1],pair2[0],pair2[1],pair3[0],pair3[1],pair4[0],pair4[1]]]
    legendtxt = ['Task 1 - {0}&{1}'.format(rulenames[0],rulenames[1]),'Task 2 - {0}&{1}'.format(rulenames[2],rulenames[3]),'Task 3 - generalise {0}&{1}'.format(rulenames[4],rulenames[5]),'Task 3 - control {0}&{1}'.format(rulenames[6],rulenames[7]), 'Task 3 - control interference']

    losses = []
    for i in range(50):
        print("random init # {0}".format(i))
        losses.append(train_generalise(len_seq,pair1,pair2,pair3,i))
    losses = (np.stack(losses,axis=2))
    with open('../results/lossgeneral_{0}_{1}_{2}.txt'.format(pair1,pair2,pair3),"wb") as fp:
        pickle.dump(losses, fp)
    print("file saved under ../results/lossgeneral_{0}_{1}_{2}.txt".format(pair1,pair2,pair3))

    nepoch,nblock,nshuf = losses.shape

    plt.figure()
    for i in range(nblock):
        df = pd.DataFrame(np.transpose(losses[:,i,:],[1,0])).melt()
        sns.lineplot(x="variable", y="value", data=df)
    plt.legend(labels=legendtxt)
    plt.ylim([0,50])
    plt.xlabel('training epochs')
    plt.ylabel('training loss')
    plt.savefig('../figures/lossgeneral_{0}_{1}_{2}.jpg'.format(pair1,pair2,pair3))
    print("figure saved under ../figures/lossgeneral_{0}_{1}_{2}.jpg".format(pair1,pair2,pair3))

#%%

if __name__ == '__main__':

    #"Force A" & "Reverse", control: "Force B"
    #ops1,ops2,ops3 = 2,1,3
    #run_generalise_1step(ops1,ops2,ops3)
    #"Force A" & "None"
    #ops1,ops2,ops3 = 2,0,3
    #run_generalise_1step(ops1,ops2,ops3)
    #"cross A" & "force A" control: "Cross B"
    #ops1,ops2,ops3 = 4,2,5
    #run_generalise_1step(ops1,ops2,ops3)
    #"force A" & "Cross A" control: "Force B"
    #ops1,ops2,ops3 = 2,4,3
    #run_generalise_1step(ops1,ops2,ops3)
    #Cross A & Force A - Force B & Reverse, test: Cross B
    pair1,pair2,pair3,pair4 = (4,2),(1,3),(4,3),(5,3)
    run_generalise_2step(pair1,pair2,pair3,pair4)

with open('../results/lossgeneral_{0}_{1}_{2}.txt'.format(pair1,pair2,pair3),"rb") as fp:
    losses = pickle.load(fp)

nepoch,nblock,nshuf = losses.shape
rulenames = [magic.dRules[i]['name'] for i in [pair1[0],pair1[1],pair2[0],pair2[1],pair3[0],pair3[1],pair4[0],pair4[1]]]
legendtxt = ['Task 1 - {0}&{1}'.format(rulenames[0],rulenames[1]),'Task 2 - {0}&{1}'.format(rulenames[2],rulenames[3]),'Task 3 - generalise {0}&{1}'.format(rulenames[4],rulenames[5]),'Task 3 - control {0}&{1}'.format(rulenames[6],rulenames[7]), 'Task 3 - control interference']

plt.figure()
for i in range(nblock):
    df = pd.DataFrame(np.transpose(losses[:,i,:],[1,0])).melt()
    sns.lineplot(x="variable", y="value", data=df)
plt.legend(labels=legendtxt)
plt.xlabel('training epochs')
plt.ylabel('training loss')
plt.ylim([0,50])
plt.savefig('../figures/lossgeneral_{0}_{1}_{2}.jpg'.format(pair1,pair2,pair3))
print("figure saved under ../figures/lossgeneral_{0}_{1}_{2}_{3}.jpg".format(pair1,pair2,pair3,pair4))
