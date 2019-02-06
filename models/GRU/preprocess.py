# obtains dictionary of words, word to index mappings....

# to be used in conjunction with the deep-learning models....

import string
import numpy as np
import torch
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def readFromFile(fname,vocabulary,translator,stop_words,ps,sent_length):
    count = 0
    sentences = []
    labels = []

    with open('../Data/'+fname+'_filtered','w') as ft:
        with open(fname) as fs:
            for line in fs:
                count+=1
                label = line[0]
                review = line[2:]
                review = review.translate(translator)
                words = word_tokenize(review.strip().lower())
                if len(words)>sent_length:
                    words = words[:sent_length]
                filtered_words = [ps.stem(w) for w in words if not w in stop_words]
                ft.write(label+','+' '.join(filtered_words))
                ft.write('\n')
                #sentences.append(words)
                #labels.append(int(label))
                for w in filtered_words:
                    vocabulary[w] = 1
                #if count%10000==0:
                 #   vocabulary = list(set(vocabulary))
                if count%100000==0:
                    print('Accessed reviews: ',count)

    return vocabulary

def word2index(**kwargs):
    w2i = {}
    count = 1
    rev_cnt = 0
    vocabulary = Counter()
    for label,fname in kwargs.items():
        with open(fname) as fs:
            for line in fs:
                rev_cnt+=1
                if rev_cnt%1000000==0:
                    print('processed reviews: ',rev_cnt)
                label = line[0]
                review = line[2:]
                words = word_tokenize(review.strip())
                for w in words:
                    vocabulary[w] = 1

    for w in vocabulary:
        w2i[w]=count
        count+=1
               
    return w2i

def obtainW2i(sent_length,**kwargs):
    translator = str.maketrans('', '', string.punctuation)
    vocabulary = Counter()
    all_sentences = {}
    all_labels = {}
    w2i = {}
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    for label,fname in kwargs.items():
        #vocabulary,sentences,labels = readFromFile(fname,vocabulary,translator)
        vocabulary = readFromFile(fname,vocabulary,translator,stop_words,ps,sent_length)
        #all_sentences[label] = sentences
        #all_labels[label] = labels
    #vocabulary,sentences_f = readFromFile(fname_f,vocabulary,translator)

    for i,w in enumerate(list(vocabulary)):
        w2i[w]=i+1

    vocabulary.clear()
    print("w2i size: ",len(w2i))
    return w2i



def encodeSentence(sentence,w2i):

    s_v = [w2i[w] for w in sentence]
    return torch.tensor(s_v, dtype=torch.long).view(-1, 1)


def testTrainSplit(sentences_m, sentences_f, sf=0.8):

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


if __name__=="__main__":

    #vocabulary,w2i,sentences_m,sentences_f = obtainW2i("../Data/sample_male","../Data/sample_female")
    #train_senetences, train_labels, test_sentences, test_labels = testTrainSplit(sentences_m,sentences_f)
    #print(test_sentences)
    #print(test_labels)
    w2i = obtainW2i(100,test = '../Data/train.csv')
