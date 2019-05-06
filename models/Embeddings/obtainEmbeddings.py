# pre compute the embeddings for the words....
# consider the set of all reviews
from gensim.models import Word2Vec

class MySentences():
    def __init__(self,fname):
        self.fname = fname

    def __iter__(self):
        for file in self.fname:
            with open(file) as fs:
                for line in fs:
                    sents = [[ for w in sent.split()] for sent in line.strip().split('.') if len(sent)>0]
                    yield sents



if __name__=="__main__":
    fname = ['../Data/train_s.csv','../Data/test_s.csv']
    sentences = MySentences(fname)
    model = Word2Vec(sentences,iter=20,size=200,workers=4) # keep min_count to default value of 5, dimension of vector =
    #  200
    model.save('amazonWord2Vec')