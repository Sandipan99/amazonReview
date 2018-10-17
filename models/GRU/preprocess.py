# obtains dictionary of words, word to index mappings....

# to be used in conjunction with the deep-learning models....

import string
import numpy as np
import torch
from collections import Counter

def readFromFile(fname,vocabulary,translator):
    count = 0
    sentences = []
    labels = []

    with open(fname) as fs:
        for line in fs:
            count+=1
            #label = line[0]
            review = line[2:]
            review = review.translate(translator)
            words = review.strip().lower().split()
            #sentences.append(words)
            #labels.append(int(label))
            for w in words:
                vocabulary[w] = 1
            #if count%10000==0:
             #   vocabulary = list(set(vocabulary))
            if count%100000==0:
                print('Accessed reviews: ',count)

    return vocabulary

def obtainW2i(**kwargs):

    translator = str.maketrans('', '', string.punctuation)
    vocabulary = Counter()
    all_sentences = {}
    all_labels = {}
    w2i = {}
    for label,fname in kwargs.items():
        #vocabulary,sentences,labels = readFromFile(fname,vocabulary,translator)
        vocabulary = readFromFile(fname,vocabulary,translator)
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

    vocabulary,w2i,sentences_m,sentences_f = obtainW2i("../Data/sample_male","../Data/sample_female")
    train_senetences, train_labels, test_sentences, test_labels = testTrainSplit(sentences_m,sentences_f)
    print(test_sentences)
    print(test_labels)
