import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.nn.utils import rnn
import torch.nn.functional as F
from Heirarchicalnet import creatingDataset,createBatches,mergeSentences,sortbylength,wordEncoder,sentenceEncoder
import sys
import itertools

import pickle
from sklearn.metrics import accuracy_score,confusion_matrix

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def inference(wordEnc,sentEnc,validation_dataset,batch_size):

    true_labels = []
    predicted_labels = []
    data = createBatches(validation_dataset,batch_size)
    for batch, lengths in data:
        if len(lengths) > 2 and len(set(lengths))==1:
            sent, label = mergeSentences(batch)
            true_labels += label
            label = torch.LongTensor(label)
            sentence_length = [len(s) for s in sent]
            sent = np.array(list(itertools.zip_longest(*sent, fillvalue=0))).T
            X = torch.from_numpy(sent)
            X_lengths = torch.LongTensor(sentence_length)
            X, X_lengths, mapped_index = sortbylength(X, X_lengths)
            batch_s = len(sentence_length)

            X, X_lengths, label = X.to(device), X_lengths.to(device), label.to(device)

            sent_out = wordEnc(X, X_lengths, batch_s)
            sent_out = sent_out.squeeze()[mapped_index, :]

            review_batch = torch.Tensor().to(device)

            r = 0
            c = sent_out.shape[1]
            for l in lengths:
                review_batch = torch.cat((review_batch, sent_out[r:r + l, :]))
                r += l
        
            
            review_batch = review_batch.view(len(lengths), -1, c)

            review_lengths = torch.LongTensor(lengths).to(device)

            output = sentEnc(review_batch, review_lengths , len(lengths))

            output = output.squeeze()

            output = F.softmax(output,dim=1)
            value,lbl = torch.max(output,1)
            predicted_labels += lbl.cpu().numpy().tolist()

            #print(true_labels)
            #print(predicted_labels)

    print(confusion_matrix(true_labels,predicted_labels))
    return accuracy_score(true_labels,predicted_labels)


if __name__=='__main__':

    with open('word2index.pickle','rb') as fs:
        w2i = pickle.load(fs)

    print('Loaded vocabulary - ',len(w2i))

    validation_dataset = creatingDataset('../Data/validation.csv',w2i)

    print('Dataset creation complete')

    w_input_size = len(w2i)
    w_encoding_size = 75
    w_hidden_size = 50
    w_output_size = 100

    s_input_size = w_output_size
    s_hidden_size = w_hidden_size
    s_repr_size = 2*w_hidden_size
    s_output_size = 2

    wordEnc = wordEncoder(w_input_size,w_encoding_size,w_hidden_size,w_output_size,0)
    sentEnc = sentenceEncoder(s_input_size,s_hidden_size,s_repr_size,s_output_size)

    wordEnc.load_state_dict(torch.load('wordEncoder_model.pt'))
    sentEnc.load_state_dict(torch.load('sentEncoder_model.pt'))

    wordEnc.to(device)
    sentEnc.to(device)

    print(inference(wordEnc,sentEnc,validation_dataset,128))
