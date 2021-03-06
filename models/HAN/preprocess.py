from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
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


def obtainKFrequentWords(k='',**kwargs): # obtain w2i without stemming or stopwords removal but only consider words with frequency at least k
    vocabulary = Counter()
    w2i = {}
    count = 0
    for label, fname in kwargs.items():
        with open(fname,errors='ignore') as fs:
            for line in fs:
                count+=1
                label = line[0]
                review = line[2:]
                words = [word.lower() for word in word_tokenize(review) if word.isalpha()]
                for w in words:
                    vocabulary[w]+=1
                if count%1000000==0:
                    print('processed reviews: ',count)            

    count = 1

    print('Original vocabulay size: ',len(vocabulary))

    for w in vocabulary:
        if vocabulary[w]>k:
            w2i[w] = count
            count+=1

    #print('Reduced vocabulary length: ',len(w2i))
    return w2i,count

def filterByFrequency(w2i,**kwargs): # filter the reviews with only words with frequency at least k and replacing others with 'unk'

    for label,fname in kwargs.items():
        with open(fname+'_filtered','w') as ft:
            with open(fname,errors='ignore') as fs:
                for line in fs:
                    label = line[0]
                    rev_or = line[2:]
                    rev_fil = '.'.join([' '.join([w if w in w2i else 'oovb' for w in word_tokenize(sent) if w.isalpha()]) for sent in sent_tokenize(rev_or.lower())])
                    ft.write(label+','+rev_fil)
                    ft.write('\n')

def filterByFrequencyIDs(w2i,**kwargs): # same function as filterByFrequency but the ids of the reviewers are present
    
    for label,fname in kwargs.items():
        with open(fname+'_filtered','w') as ft:
            with open(fname) as fs:
                for line in fs:
                    line = line.strip() 
                    x = line.find(',')
                    id_ = line[:x]
                    label = line[-1]
                    rev_or = line[x+1:-2]
                    rev_fil = '.'.join([' '.join([w if w in w2i else 'oovb' for w in word_tokenize(sent) if w.isalpha()]) for sent in sent_tokenize(rev_or.lower())])
                    ft.write(id_+','+rev_fil+','+label)
                    ft.write('\n')

if __name__ == '__main__':

    w2i,count = obtainKFrequentWords(k=5,train_file='/home/rachneet/datasets/reddit/train_reddit.csv', validation_file='/home/rachneet/datasets/reddit/validation_reddit.csv')
    print('vocabulary size - ',len(w2i))
 
    print(len(w2i),count)

    filterByFrequency(w2i,train_file='/home/rachneet/datasets/reddit/train_reddit.csv', validation_file='/home/rachneet/datasets/reddit/validation_reddit.csv')

    w2i['oovb'] = count
    
    with open('word2index.pickle','wb') as ft:
       pickle.dump(w2i,ft)
 
