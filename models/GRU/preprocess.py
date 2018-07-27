# obtains dictionary of words, word to index mappings....

# to be used in conjunction with the deep-learning models....

import string
import numpy as np
import torch

def readFromFile(fname,vocabulary,translator):

    sentences = []
    with open(fname) as fs:
        for line in fs:
            line = line.translate(translator)
            words = line.strip().lower().split()
            sentences.append(words)
            for w in words:
                vocabulary.append(w)

    return vocabulary,sentences

def obtainW2i(fname_m,fname_f):

    translator = str.maketrans('', '', string.punctuation)
    vocabulary = []
    w2i = {}
    w_c = 0
    vocabulary,sentences_m = readFromFile(fname_m,vocabulary,translator)
    vocabulary = list(set(vocabulary))
    vocabulary,sentences_f = readFromFile(fname_f,vocabulary,translator)
    vocabulary = list(set(vocabulary))

    for i in range(len(vocabulary)):
        w2i[vocabulary[i]] = i

    return vocabulary,w2i,sentences_m,sentences_f

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
