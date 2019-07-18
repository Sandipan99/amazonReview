import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.nn.utils import rnn
import torch.nn.functional as F
from HierarchicalAttentionNet_pre_embed import createBatches,sortbylength,wordEncoder,sentenceEncoder,text2tensor,createEmbeddingMatrix
import sys
import itertools
import gensim

import pickle
from sklearn.metrics import accuracy_score,confusion_matrix

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def creatingDatasetIDs(fname, w2i, max_length=15):
    dataset={}
    with open(fname + '_filtered') as fs:
        for line in fs:
            line = line.strip()
            x = line.find(',')
            id_ = line[:x]
            label = int(line[-1])
            review = line[x+1:-2]
            temp = review.split('.')
            encoded_review = text2tensor(temp, w2i)
            length = len(encoded_review)
            if length>max_length:
                continue
            if length not in dataset and length > 0:
                dataset[length] = []
            if length > 0:
                dataset[length].append((encoded_review, label, id_))
    return dataset

def mergeSentences(batch):
    sent = []
    label = []
    rev_id = []
    for review, l, id_ in batch:
        sent += review
        label.append(l)
        rev_id.append(id_)
    return sent, label, rev_id

def inference(wordEnc,sentEnc,validation_dataset,batch_size):

    wordEnc.to(device)
    sentEnc.to(device)

    wordEnc.eval()
    sentEnc.eval()

    true_labels = []
    predicted_labels = []
    review_id = []
    data = createBatches(validation_dataset,batch_size)

    ft = open('output_han_amazon.csv','w')
    ft.write('True_label,Predicted_label,ReviewerID')
    ft.write('\n')

    with torch.no_grad():
        for batch, lengths in data:
            if len(lengths) > 2 and len(set(lengths))==1:
                sent, label, id_ = mergeSentences(batch)
                true_labels += label
                review_id += id_
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
    
    for t_l,p_l,id_ in zip(true_labels,predicted_labels,review_id):
        ft.write(str(t_l)+','+str(p_l)+','+str(id_))
        ft.write('\n')

    ft.close()

    print(confusion_matrix(true_labels,predicted_labels))
    return accuracy_score(true_labels,predicted_labels)


if __name__=='__main__':

    with open('word2index.pickle','rb') as fs:
        w2i = pickle.load(fs)

    print('Loaded vocabulary - ',len(w2i))

    if os.path.exits(test_file+'_filtered'):
        print('filtered file already exists... skipping creation of filtered file')
    else:
        print('filtered file not found... creating filtered file')
        pp.filterByFrequencyIDs(w2i,test_file=test_file)




    test_dataset = creatingDatasetIDs('../../../amazonUser/User_level_test_with_id.csv',w2i)

    print('Dataset creation complete')

    model = gensim.models.Word2Vec.load('../Embeddings/amazonWord2Vec')

    w_input_size,w_encoding_size = model.wv.vectors.shape

    matrix = createEmbeddingMatrix(model,w2i,w_encoding_size)

    print('embedding matrix obtained.')

    #w_input_size = len(w2i)
    #w_encoding_size = 75
    w_hidden_size = 50
    w_output_size = 100

    s_input_size = w_output_size
    s_hidden_size = w_hidden_size
    s_repr_size = 2*w_hidden_size
    s_output_size = 2

    padding_idx = 0

    wordEnc = wordEncoder(matrix,w_input_size+1,w_encoding_size,w_hidden_size,w_output_size,padding_idx)
    sentEnc = sentenceEncoder(s_input_size,s_hidden_size,s_repr_size,s_output_size)

    wordEnc.load_state_dict(torch.load('wordEncoder_model-pre_embed.pt'))
    sentEnc.load_state_dict(torch.load('sentEncoder_model-pre_embed.pt'))

    print(inference(wordEnc,sentEnc,test_dataset,128))
