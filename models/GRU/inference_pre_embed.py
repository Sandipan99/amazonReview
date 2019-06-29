from torch.utils import data
import torch
import torch.nn as nn
import numpy as np

from models.GRU.RNN_model_batch_pre_embed import Encoder
from models.GRU.inference import DatasetInfer,encodeDatasetIds
from models.HAN import HierarchicalAttentionNet_pre_embed as hp
import pickle
import gensim

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

    with open('inference_preembed_result.csv','w') as ft:
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


model = gensim.models.Word2Vec.load('../Embeddings/amazonWord2Vec')

input_size,encoding_size = model.wv.vectors.shape

matrix =hp.createEmbeddingMatrix(model,w2i,encoding_size)

input_size, encoding_size = model.wv.vectors.shape

hidden_size = 250
output_size = 2
layers = 2
batch_size = 256

encoder = Encoder(matrix,input_size+1,encoding_size, hidden_size, output_size,layers, padding_idx)
encoder.load_state_dict(torch.load('models_amazon/RNN_pretrain.pt'))
encoder = encoder.to(device)

print('model loaded')

dataset_test = DatasetInfer(reviews_test,labels_test,lengths_test,ids_test)
inference(encoder,dataset_test,batch_size)
