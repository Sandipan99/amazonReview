# Do not use this code... use RNN model batch instead....

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import preprocess as pp
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.LogSigmoid()

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded.view(1,1,-1) # view is same as reshape
        output, hidden = self.gru(output, hidden)
        output = self.sigmoid(self.out(output[0]))
        return output, hidden

    def initHidden(self):
         return torch.zeros(1, 1, self.hidden_size)

def findTrainExample(train_senetences, train_labels):
    ind = random.randint(0,len(train_senetences)-1)
    return train_senetences[ind], train_labels[ind]



def train(encoder, train_senetences, train_labels, w2i, iter=1, learning_rate=0.001):

    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    for i in range(iter):
        sentence, label = findTrainExample(train_senetences, train_labels)
        sentence_tensor = pp.encodeSentence(sentence,w2i)
        label_tensor = torch.tensor(label,dtype=torch.float32).view(1,1)

        encoder_hidden = encoder.initHidden()

        input_length = sentence_tensor.size(0)
        for ei in range(input_length):
            output, encoder_hidden = encoder(sentence_tensor[ei],encoder_hidden)

        loss = torch.abs(output - label_tensor)
        if i%100==0:
            print("loss: ",loss)

        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        optimizer.step()


def findBatch(train_sentences, train_labels, batch_size):

    batch_sentences = []
    batch_labels = []
    train_size = len(train_labels)

    while batch_size>0:
        ind = np.random.randint(0,int(train_size))
        batch_sentences.append(train_sentences[ind])
        batch_labels.append(train_labels[ind])
        batch_size-=1
    return batch_sentences,batch_labels


def batch_train(encoder, train_sentences, train_labels, batch_size, w2i, epochs=5, learning_rate=0.001):
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    for i in range(1000):
        batch_sentences, batch_labels = findBatch(train_sentences, train_labels, batch_size)
        print('found batch')
        loss = 0
        encoder_hidden = encoder.initHidden()
        for j in range(len(batch_sentences)):
            sentence = batch_sentences[j]
            label = batch_labels[j]
            sentence_tensor = pp.encodeSentence(sentence,w2i)
            label_tensor = torch.tensor(label,dtype=torch.float32).view(1,1)

            input_length = sentence_tensor.size(0)
            for ei in range(input_length):
                output, encoder_hidden = encoder(sentence_tensor[ei],encoder_hidden)

            loss += torch.abs(output - label_tensor)
            #print(loss)
        loss = loss/len(batch_sentences)
        print(loss)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()


def evaluate(encoder, test_sentences, test_labels, w2i):

    accuracy = 0.0

    for i in range(len(test_sentences)):
        sentence = test_sentences[i]
        label = test_labels[i]
        sentence_tensor = pp.encodeSentence(sentence,w2i)
        label_tensor = torch.tensor(label,dtype=torch.float32).view(1,1)

        encoder_hidden = encoder.initHidden()

        input_length = sentence_tensor.size(0)
        for ei in range(input_length):
            output, encoder_hidden = encoder(sentence_tensor[ei],encoder_hidden)

        output = torch.round(output)
        if torch.equal(output,label_tensor):
            accuracy+=1

    return accuracy/len(test_sentences)

if __name__=='__main__':

    # file name for male reviews and female reviews
    vocabulary,w2i,sentences_m,sentences_f = pp.obtainW2i("../Data/sample_male","../Data/sample_female")
    train_senetences, train_labels, test_sentences, test_labels = pp.testTrainSplit(sentences_m, sentences_f)
    hidden_size = 20
    input_size = len(w2i)
    output_size = 1
    encoder = Encoder(input_size, hidden_size, output_size)
    batch_train(encoder, train_senetences, train_labels, 3, w2i)
    #accuracy = evaluate(encoder,test_sentences, test_labels,w2i)
    #print(accuracy)
