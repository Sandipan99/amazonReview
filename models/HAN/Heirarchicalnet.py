import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.nn.utils import rnn

from preprocess import clean
import itertools

import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def text2tensor(review,w2i):
    out = [[w2i[w] for w in sents.split()] for sents in review if len(sents)>0]
    return out

def creatingDataset(fname,w2i):  # dictionary of list of tuples (rev,label)
    dataset = {}
    with open(fname+'_cleaned') as fs:
        for line in fs:
            label = int(line[0])
            review = line[2:]
            temp = review.strip().split('.')
            length = len(temp)
            if length not in dataset:
                dataset[length] = []
            encoded_review = text2tensor(temp,w2i)
            dataset[length].append((encoded_review,label))
    return dataset


def createBatches(dataset, batch_size):  # generator implementation
    batch = []  # return a batch of datapoints based on batch_size
    lengths = list(dataset.keys())
    lengths.sort(reverse=True)
    size = 0
    sent_length = []
    for l in lengths:
        for doc in dataset[l]:
            if len(batch) > 0:
                curr_len = len(batch[-1][0])
            else:
                curr_len = l
            diff = curr_len - len(doc[0])
            if diff <= 2 and diff > 0:
                size += 1
                batch.append(doc)
                sent_length.append(len(doc[0]))
            elif diff == 0:
                batch.append(doc)
                sent_length.append(len(doc[0]))
                size += 1
            else:
                if len(batch) > 0:
                    yield (batch, sent_length)
                batch = [doc]
                sent_length = [len(doc[0])]
                size = 1

            if size == batch_size:
                yield (batch, sent_length)
                batch = []
                sent_length = []
                size = 0
    yield (batch, sent_length)

def mergeSentences(batch):
    sent = []
    label = []
    for review,l in batch:
        sent+=review
        label.append(l)
    return sent,label


class wordEncoder(nn.Module):
    def __init__(self, input_size, encoding_size, hidden_size, output_size, padding_idx):
        super(wordEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = True

        self.embedding = nn.Embedding(input_size, encoding_size, padding_idx=padding_idx)
        self.e2i = nn.Linear(encoding_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.h2o = nn.Linear(2 * hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.u_w = nn.Parameter(torch.rand(output_size))  # word context
        self.softmax = nn.Softmax(dim=0)

    def forward(self, X, X_lengths, batch_size):
        self.hidden = self.initHidden(batch_size)
        X = self.embedding(X)
        X = self.e2i(X)

        X = rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        X, self.hidden = self.gru(X, self.hidden)

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        H = torch.unbind(X, dim=0)
        X = self.h2o(X)
        X = self.tanh(X)

        Y = torch.unbind(X, dim=0)

        Y_1 = torch.Tensor()
        for i in range(X_lengths.shape[0]):
            x = self.softmax(torch.sum(Y[i][:X_lengths[i].item()] * self.u_w, dim=1)).view(-1, 1)
            Y_1 = torch.cat((Y_1, torch.sum(H[i][:X_lengths[i].item()] * x, dim=0).view(1, 1, -1)), dim=1)

        return Y_1

    def initHidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size)


class sentenceEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, repr_size, output_size):
        super(sentenceEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.g2r = nn.Linear(2 * hidden_size, repr_size)
        self.tanh = nn.Tanh()
        self.u_s = nn.Parameter(torch.rand(repr_size))  # sentence context
        self.softmax = nn.Softmax(dim=0)
        self.r2o = nn.Linear(2 * hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, X_lengths, batch_size):
        self.hidden = self.initHidden(batch_size)

        X = rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        X, self.hidden = self.gru(X, self.hidden)

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        H = torch.unbind(X, dim=0)  # hidden state obtained from each sentence

        X = self.g2r(X)
        X = self.tanh(X)

        Y = torch.unbind(X, dim=0)

        Y_1 = torch.Tensor()
        for i in range(X_lengths.shape[0]):
            x = self.softmax(torch.sum(Y[i][:X_lengths[i].item()] * self.u_s, dim=1)).view(-1, 1)
            Y_1 = torch.cat((Y_1, torch.sum(H[i][:X_lengths[i].item()] * x, dim=0).view(1, 1, -1)), dim=1)

        output = self.r2o(Y_1)
        output = self.sigmoid(output)

        return output

    def initHidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size)


# concatenating sentences from all reviews in a batch and then sorting them based on length would change the order
# to keep track of the sequence of the sentences we need to remember the original mapping
# this routine keeps track of the sequence of the sentences among all the reviews
# need when passing to the sentence encoder...

def originalMap(indices):
    count = 0
    m = {}
    for i in indices:
        m[i.item()] = count
        count+=1

    m_p = []
    for i,val in sorted(m.items(),key=lambda x:x[0]):
        m_p.append(val)

    return torch.LongTensor(m_p)

def sortbylength(all_sent,all_sent_len):
    sorted_lengths, indices = torch.sort(all_sent_len,descending=True)
    mapped_index = originalMap(indices)
    return all_sent[torch.LongTensor(indices),:],sorted_lengths,mapped_index


def train(wordEnc, sentEnc, train_dataset, batch_size=64, epochs=10, learning_rate=0.001):
    wordEnc_optimizer = optim.Adam(wordEnc.parameters(), lr=learning_rate)
    sentEnc_optimizer = optim.Adam(sentEnc.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for batch, lengths in createBatches(train_dataset, batch_size):
            if len(lengths) < 16:
                continue
            sent, label = mergeSentences(batch, lengths)
            label = torch.LongTensor(label)
            sentence_length = [len(s) for s in sent]
            sent = np.array(list(itertools.zip_longest(*sent, fillvalue=0))).T
            X = torch.from_numpy(sent)
            X_lengths = torch.LongTensor(sentence_length)
            X, X_lengths, mapped_index = sortbylength(X, X_lengths)
            batch_size = len(sentence_length)

            X, X_lengths, label = X.to(device), X_lengths.to(device), label.to(device)

            sent_out = wordEnc(X, X_lengths, batch_size)
            sent_out = sent_out.squeeze()[mapped_index, :]

            curr_length = lengths[0]

            review_batch = torch.Tensor()

            r = 0
            c = sent_out.shape[1]
            for l in lengths:
                if l == curr_length:
                    review_batch = torch.cat((review_batch, sent_out[r:r + l, :]))
                    r += l
                else:
                    diff = curr_length - l
                    review_batch = torch.cat((review_batch, sent_out[r:r + l, :], torch.zeros(diff, c)))
                    r += l

            review_batch = review_batch.view(len(lengths), -1, c)

            review_batch = review_batch.to(device)

            review_lengths = torch.LongTensor(lengths).to(device)

            output = sentEnc(review_batch, review_lengths , len(lengths))

            loss = criterion(output.squeeze(), label)

            print(loss)

            wordEnc_optimizer.zero_grad()
            sentEnc_optimizer.zero_grad()
            loss.backward()
            sentEnc_optimizer.step()
            wordEnc_optimizer.step()


if __name__=='__main__':

    with open('word2index.pickle','rb') as fs:
        w2i = pickle.load(fs)

    train_dataset = creatingDataset('../Data/test_s.csv', w2i)
    validation_dataset = creatingDataset('../Data/validation_s.csv',w2i)

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

    wordEnc.to(device)
    sentEnc.to(device)

    train(wordEnc,sentEnc,train_dataset)

