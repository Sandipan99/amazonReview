# simple one layer RNN model....

# use preprocess.py for pre-processing tasks..

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import preprocess as pp

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.LogSigmoid()

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1,1,-1) # view is same as reshape
        output = embedded
        output, hidden = self.gru(output, hidden)
        output = self.sigmoid(self.out(output[0]))
        return output, hidden

    def initHidden(self):
         return torch.zeros(1, 1, self.hidden_size)

def findTrainExample(train_senetences, train_labels):
    ind = random.randint(0,len(train_senetences)-1)
    return train_senetences[ind], train_labels[ind]



def train(encoder, iter=5000, learning_rate=0.001, train_senetences, train_labels, w2i):

    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    for i in range(iter):
        sentence, label = findTrainExample(train_senetences, train_labels)
        sentence_tensor = encodeSentence(sentence,w2i)
        label_tensor = torch.tensor(label,dtype=torch.float32).view(1,1)

        encoder_hidden = encode.initHidden()

        input_length = sentence_tensor.size(0)
        for ei in range(input_length):
            output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)

        loss = torch.abs(output - label_tensor)

        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        optimizer.step()

def evaluate(encoder, test_sentences, test_labels, w2i):

    accuracy = 0.0

    for i in range(len(test_sentences)):
        sentence = test_sentences[i]
        label = test_labels[i]
        sentence_tensor = encodeSentence(sentence,w2i)
        label_tensor = torch.tensor(label,dtype=torch.float32).view(1,1)

        encoder_hidden = encode.initHidden()

        input_length = sentence_tensor.size(0)
        for ei in range(input_length):
            output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)

        output = torch.round(output)
        if torch.equal(output,label_tensor):
            accuracy+=1

    return accuracy/len(test_sentences)

if __name__=='__main__':

    # file name for male reviews and female reviews
    vocabulary,w2i,sentences_m,sentences_f = pp.obtainW2i(fname_m,fname_f)
    train_senetences, train_labels, test_sentences, test_labels = pp.testTrainSplit(sentences_m, sentences_f)
    hidden_size = 20
    input_size = len(w2i)
    output_size = 1
    encoder = Encoder(input_size, hidden_size, output_size)
    train(encoder)                
