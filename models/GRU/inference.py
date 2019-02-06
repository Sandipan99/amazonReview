from torch.utils import data
from torch.nn.utils import rnn
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import preprocess as pp
import numpy as np

from RNN_model_batch import Encoder,Dataset
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sentence2tensor(sentence,w2i,pad,sent_length):
    S = [w2i[w] for w in sentence]
    if len(S)>sent_length:
        return S[:sent_length]
    else:
        x = len(S)
        S = S + [pad for i in range(sent_length - x)]
        return S

def encodeDataset(fname,w2i,padding_idx,sent_length):
    reviews = []
    labels = []
    lengths = []
    count = 0
    with open(fname+'_filtered') as fs:
         for line in fs:
             count+=1
             label = line[0]
             review = line[2:]
             words = word_tokenize(review.strip())
             if len(words)>0:
                reviews.append(sentence2tensor(words,w2i,padding_idx,sent_length))
                if len(words)>sent_length:
                    lengths.append(sent_length)
                else:
                    lengths.append(len(words))
                labels.append(int(label))
                if count%100000==0:
                    print('Encoded reviews: ',count)


    return reviews,labels,lengths

def sortbylength(X,y,s_lengths):
    sorted_lengths, indices = torch.sort(s_lengths,descending=True)
    return X[torch.LongTensor(indices),:],y[torch.LongTensor(indices)],sorted_lengths

def inference(encoder,dataset_test,batch_size):
    accuracy = []
    count = 0
    loader = data.DataLoader(dataset_test,batch_size=batch_size)
    for X,y,X_lengths in loader:
        count+=1
        if count%100==0:
            print('calculated batches: ',count)
        X,y,X_lengths = sortbylength(X,y,X_lengths)
        X,y,X_lengths = X.to(device),y.to(device),X_lengths.to(device)
        b_size = y.size(0)
        output = encoder(X,X_lengths,b_size)
        output = F.softmax(output,dim=1)
        value,labels = torch.max(output,1)

        accuracy.append(accuracy_score(y.cpu().numpy(),labels.cpu().numpy()))
    
    return np.mean(accuracy)


with open('word2index','rb') as fs:
        w2i = pickle.load(fs)

print('loaded vocabulary')
print('size of vocabulary: ',len(w2i))
sent_length = 100
test_file = '../Data/test.csv'
vocab_size = len(w2i)
padding_idx = 0
hidden_size = 250
input_size = vocab_size
output_size = 2
layers = 1
batch_size = 256
encoding_size = 50
encoder = Encoder(input_size,encoding_size, hidden_size, output_size,layers, padding_idx)
encoder.load_state_dict(torch.load('encoder_model.pt'))
encoder = encoder.to(device)

print('model loaded')

reviews_test, labels_test, lengths_test = encodeDataset(test_file,w2i,padding_idx,sent_length)

dataset_test = Dataset(reviews_test,labels_test,lengths_test)
print(inference(encoder,dataset_test,batch_size))
