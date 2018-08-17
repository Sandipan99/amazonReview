# simple one layer RNN model....

# use preprocess.py for pre-processing tasks..

# improve on this

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
import string

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#preprocessing
def readFromFile(fname,vocabulary,translator):

    count = 0
    sentences = []
    with open(fname) as fs:
        for line in fs:
            count +=1
            line = line.translate(translator)
            words = line.strip().lower().split()
            sentences.append(words)
            for w in words:
                vocabulary.append(w)

            if(count%50==0):
                vocabulary = list(set(vocabulary))
                
            if count%1000==0:
                print("read: ",count)
                
    vocabulary = list(set(vocabulary))

    return vocabulary,sentences

def obtainW2i(fname_m,fname_f):

    translator = str.maketrans('', '', string.punctuation)
    vocabulary = []
    w2i = {}
    w_c = 0
    vocabulary,sentences_m = readFromFile(fname_m,vocabulary,translator)
    #vocabulary = list(set(vocabulary))
    vocabulary,sentences_f = readFromFile(fname_f,vocabulary,translator)
    #vocabulary = list(set(vocabulary))

    for i in range(len(vocabulary)):
        w2i[vocabulary[i]] = i

    return vocabulary,w2i,sentences_m,sentences_f


def encodeSentence(sentence,w2i):

    s_v = [w2i[w] for w in sentence]
    return torch.tensor(s_v, dtype=torch.long).view(-1, 1)


def testTrainSplit(sentences_m, sentences_f, sf=0.75):

    train_senetences = []
    train_labels = []

    test_sentences = []
    test_labels = []

    for s in sentences_m:
        if np.random.uniform() < sf:
            train_senetences.append(s)
            train_labels.append(1)
        else:
            test_sentences.append(s)
            test_labels.append(1)

    for s in sentences_f:
        if np.random.uniform() < sf:
            train_senetences.append(s)
            train_labels.append(0)
        else:
            test_sentences.append(s)
            test_labels.append(0)

    return train_senetences, train_labels, test_sentences, test_labels


#end preprocessing
#=================

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        #self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        for ei in range(input.size(0)):
            embedded = self.embedding(input[ei])
            output = embedded.view(1,1,-1) # view is same as reshape
            output, hidden = self.gru(output, hidden)
            #print(output)
            #output = self.relu(self.out(output[0]))
            #print(output)
            output = self.sigmoid(self.out(output[0]))
            #print(output)
        return output

    def initHidden(self):
         return torch.zeros(1, 1, self.hidden_size)

def findTrainExample(train_senetences, train_labels):
    ind = random.randint(0,len(train_senetences)-1)
    return train_senetences[ind], train_labels[ind]



def train(encoder, train_senetences, train_labels, w2i, iter=1, learning_rate=0.001):

    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    for i in range(iter):
        sentence, label = findTrainExample(train_senetences, train_labels)
        sentence_tensor = encodeSentence(sentence,w2i)
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
        if len(train_sentences[ind])>0:
            batch_sentences.append(train_sentences[ind])
            batch_labels.append(train_labels[ind])
            batch_size-=1
    return batch_sentences,batch_labels


def batch_train(encoder, train_sentences, train_labels, batch_size, w2i, epochs=5, learning_rate=0.001):
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    for i in range(1500):
        #print("encoder weight: ",encoder.out.weight)
        batch_sentences, batch_labels = findBatch(train_sentences, train_labels, batch_size)
        #print('found batch')
        loss = 0
        #encoder_hidden = encoder.initHidden()
        for j in range(len(batch_sentences)):
            sentence = batch_sentences[j]
            label = batch_labels[j]
            sentence_tensor = encodeSentence(sentence,w2i)
            label_tensor = torch.tensor(label,dtype=torch.float32).view(1,1)
            encoder_hidden = encoder.initHidden()
            encoder_hidden = encoder_hidden.to(device)
            #input_length = sentence_tensor.size(0)
            #for ei in range(input_length):
            sentence_tensor = sentence_tensor.to(device)
            label_tensor = label_tensor.to(device)
            output = encoder(sentence_tensor,encoder_hidden)

            loss += torch.mul((output - label_tensor),(output - label_tensor))
            #print(loss)
        loss = loss/len(batch_sentences)
        if(i%100 == 0):
            print(loss)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()


def evaluate(encoder, test_sentences, test_labels, w2i):

    accuracy = 0.0
    #print("encoder weight: ",encoder.out.weight)
    for i in range(len(test_sentences)):
        sentence = test_sentences[i]
        label = test_labels[i]
        sentence_tensor = encodeSentence(sentence,w2i)
        label_tensor = torch.tensor(label,dtype=torch.float32).view(1,1)

        encoder_hidden = encoder.initHidden()
        encoder_hidden = encoder_hidden.to(device)
        #input_length = sentence_tensor.size(0)
        #for ei in range(input_length):
        sentence_tensor = sentence_tensor.to(device)
        label_tensor = label_tensor.to(device)
        output = encoder(sentence_tensor,encoder_hidden)
        #print("output from encoder: ",output)
        #output = torch.abs(output)
        output = torch.round(output)
        #print("output: ",output)
        #print("label: ",label_tensor)
        if torch.equal(output,label_tensor):
            accuracy+=1
        
        #print ("accuracy: ",accuracy)

    return accuracy/len(test_sentences)

if __name__=='__main__':

    # file name for male reviews and female reviews
    vocabulary,w2i,sentences_m,sentences_f = obtainW2i("/home/rachneet/sample_male","/home/rachneet/sample_female")
    print(len(vocabulary),len(w2i))
    
    train_sentences, train_labels, test_sentences, test_labels = testTrainSplit(sentences_m, sentences_f)
    print("train set: ",len(train_sentences),"test set: ",len(test_sentences))
    
    #validation set  
    validation_sentences = []
    validation_labels = []

    for i in range(5000):

        validation_sentences.append(test_sentences[i])
        validation_labels.append(test_labels[i])
    
    hidden_size = 100
    input_size = len(w2i)
    output_size = 1
    encoder = Encoder(input_size, hidden_size, output_size)
    encoder = encoder.to(device)
    #batch_train(encoder, train_sentences, train_labels, 50, w2i)
    #print("train_complete............")
    
    j=1
    for epoch in range(5):
        batch_train(encoder, train_senetences, train_labels,50, w2i)
        result = evaluate(encoder,validation_sentences, validation_labels,w2i)
        print("Result after epoch {} : {}".format(j,result))
        j+=1
    
    print("Train Complete")
    
    accuracy = evaluate(encoder,test_sentences, test_labels,w2i)
    print("Model Accuracy:",accuracy)
