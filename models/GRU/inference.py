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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class DatasetInfer(data.Dataset):
    def __init__(self, reviews, labels, lengths, ids_):
        self.reviews = reviews
        self.labels = labels
        self.lengths = lengths
        self.ids_ = ids_

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self,index):
        X = torch.tensor(self.reviews[index],dtype=torch.long)
        y = torch.tensor(self.labels[index],dtype=torch.long)
        s_lengths = self.lengths[index]
        r_ids = self.ids_[index]
        return X,y,s_lengths,r_ids


def sentence2tensor(sentence,w2i,pad,sent_length):
    S = [w2i[w] for w in sentence]
    if len(S)>sent_length:
        return S[:sent_length]
    else:
        x = len(S)
        S = S + [pad for i in range(sent_length - x)]
        return S

def encodeDatasetIDs(fname,w2i,padding_idx,sent_length):
    reviews = []
    labels = []
    lengths = []
    ids_ = []

    count = 0
    with open(fname+'_filtered') as fs:
         for line in fs:
             line = line.strip()
             count+=1
             label = line[-1]
             ind = line.find(',')
             id_ = line[:ind]
             review = line[ind+1:-2]
             words = word_tokenize(review)
             words = [w for w in words if w.isalpha()]
             if len(words)>0:
                reviews.append(sentence2tensor(words,w2i,padding_idx,sent_length))
                if len(words)>sent_length:
                    lengths.append(sent_length)
                else:
                    lengths.append(len(words))
                labels.append(int(label))
                ids_.append(id_)
                if count%100000==0:
                    print('Encoded reviews: ',count)


    return reviews,labels,lengths,ids_

def sortbylengthIDs(X,y,s_lengths,ids_):
    sorted_lengths, indices = torch.sort(s_lengths,descending=True)
    ind = indices.numpy()
    ids_ = [ids_[i] for i in ind]
    return X[torch.LongTensor(indices),:],y[torch.LongTensor(indices)],sorted_lengths,ids_

def inference(encoder,dataset_test,batch_size):

    encoder.to(device)
    encoder.eval()

    
    true_labels = []
    predicted_labels = []
    reviewer_id = []
    with torch.no_grad():
        loader = data.DataLoader(dataset_test,batch_size=batch_size)
        for X,y,X_lengths,ids_ in loader:
            X,y,X_lengths,ids_ = sortbylengthIDs(X,y,X_lengths,ids_)
            X,y,X_lengths = X.to(device),y.to(device),X_lengths.to(device)
            b_size = y.size(0)
            output = encoder(X,X_lengths,b_size)
            output = F.softmax(output,dim=1)
            value,labels = torch.max(output,1)

            true_labels+=y.cpu().numpy().tolist()
            predicted_labels+=labels.cpu().numpy().tolist()
            reviewer_id+=ids_

    with open('inference_result.csv','w') as ft:
        ft.write('True_label,Predicted_label,ReviewerID\n')
        for t_l,p_l,rev_id in zip(true_labels,predicted_labels,reviewer_id):
            ft.write(str(t_l)+','+str(p_l)+','+rev_id)
            ft.write('\n')


with open('../HAN/word2index.pickle','rb') as fs:
        w2i = pickle.load(fs)

print('loaded vocabulary')
print('size of vocabulary: ',len(w2i))
sent_length = 100
test_file = '../../../amazonUser/User_level_test_with_id.csv'
vocab_size = len(w2i)
padding_idx = 0

reviews_test, labels_test, lengths_test, ids_test = encodeDatasetIDs(test_file,w2i,padding_idx,sent_length)

hidden_size = 250
input_size = vocab_size
output_size = 2
layers = 2
batch_size = 256
encoding_size = 50
encoder = Encoder(input_size,encoding_size, hidden_size, output_size,layers, padding_idx)
encoder.load_state_dict(torch.load('models_amazon/RNN_vanilla_2.pt'))
encoder = encoder.to(device)

print('model loaded')

dataset_test = DatasetInfer(reviews_test,labels_test,lengths_test,ids_test)
inference(encoder,dataset_test,batch_size)
