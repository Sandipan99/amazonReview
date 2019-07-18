# a batch implementation of basic RNN model
# vocab is filtered based on the frequency of words... only use words with frequency at least 5 ... replace others with unk token
# pre-trained embeddings used and weights not updated during training

from torch.utils import data
from torch.nn.utils import rnn
from sklearn.metrics import accuracy_score

import string
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import preprocess as pp
import numpy as np

from models.HAN import HierarchicalAttentionNet_pre_embed as hp
from models.GRU import RNN_model_batch as rm
from models.GRU.RNN_model_batch import Dataset
import pickle
import gensim

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, matrix, input_size, encoding_size, hidden_size, output_size, layers, padding_idx):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.layers = layers
        self.embedding = nn.Embedding(input_size, encoding_size, padding_idx=padding_idx)
        self.embedding.load_state_dict({'weight': matrix})
        self.embedding.weight.requires_grad = False

        self.e2i = nn.Linear(encoding_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=self.layers)
        self.dense1 = nn.Linear(hidden_size,hidden_size)
        self.dense2 = nn.Linear(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.batch_first = True

    def forward(self, X, X_lengths, batch_size):

        self.hidden = self.initHidden(batch_size)
        X = self.embedding(X)
        X = self.e2i(X)
        X = rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        X, self.hidden = self.gru(X, self.hidden)

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        idx = (torch.cuda.LongTensor(X_lengths) - 1).view(-1, 1).expand(len(X_lengths), X.size(2))

        time_dimension = 1 if self.batch_first else 0
        idx = idx.unsqueeze(time_dimension)
        X = X.gather(time_dimension, Variable(idx)).squeeze(time_dimension)

        X = self.dense1(X)
        X = self.dense2(X)

        X = self.out(X)
        X = self.sigmoid(X)

        return X

    def initHidden(self,batch_size):
         return torch.zeros(self.layers, batch_size, self.hidden_size).to(device)


if __name__=='__main__':

    sent_length = 100
    train_file,validation_file = '../../../amazonUser/User_level_train.csv','../../../amazonUser/User_level_validation.csv'
    with open('../HAN/word2index.pickle','rb') as fs:
        w2i = pickle.load(fs)

    print('loaded vocabulary')
    print('size of vocabulary: ',len(w2i))

    vocab_size = len(w2i)
    padding_idx = 0 

    reviews_train,labels_train,lengths_train = rm.encodeDataset(train_file,w2i,padding_idx,sent_length)
    reviews_validate, labels_validate, lengths_validate = rm.encodeDataset(validation_file,w2i,padding_idx,sent_length)
    #reviews_test, labels_test, lengths_test = encodeDataset(test_file,w2i,padding_idx,sent_length)

    
    print('created batches from data loader')

    model = gensim.models.Word2Vec.load('../Embeddings/amazonWord2Vec')

    input_size,encoding_size = model.wv.vectors.shape

    matrix =hp.createEmbeddingMatrix(model,w2i,encoding_size)

    input_size, encoding_size = model.wv.vectors.shape
    #print(reviews[2])
    dataset_train = Dataset(reviews_train,labels_train,lengths_train)
    dataset_validate = Dataset(reviews_validate,labels_validate,lengths_validate)

    hidden_size = 250
    output_size = 2
    layers = 2
    batch_size = 256
    encoder = Encoder(matrix, input_size+1, encoding_size, hidden_size, output_size,layers, padding_idx)
    encoder = encoder.to(device)
    rm.train(encoder,dataset_train, dataset_validate, batch_size,saveas='models_amazon/RNN_pretrain.pt')


