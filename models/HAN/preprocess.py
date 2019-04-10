from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import pickle

def clean(**kwargs):  # obtains a word to index mapping as well as clean the dataset of punctuation and stopwords
    count = 0
    translator = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    vocabulary = Counter()
    w2i = {}
    for label, fname in kwargs.items():
        with open(fname + '_cleaned', 'w') as ft:
            with open(fname) as fs:
                for line in fs:
                    count += 1
                    label = line[0]
                    review = line[2:]
                    sents = review.strip().split('.')
                    sents = [[ps.stem(w) for w in word_tokenize(s.translate(translator)) if w not in stop_words] \
                             for s in sents if len(s) > 1]
                    words = sum(sents, [])
                    for w in words:
                        vocabulary[w] = 1
                    rev = '.'.join([' '.join(s) for s in sents])
                    ft.write(label + ',' + rev)
                    ft.write('\n')
                    if count % 1000000 == 0:
                        print('cleaned reviews: ', count)

    count = 1
    for w in vocabulary:
        w2i[w] = count
        count += 1
    return w2i


if __name__ == '__main__':

    w2i = clean(train_file='../Data/train.csv', validation_file='../Data/validation.csv', test_file='../Data/test.csv')
    print('vocabulary size - ',len(w2i))
    with open('word2index.pickle','wb') as ft:
        pickle.dump(w2i,ft)
