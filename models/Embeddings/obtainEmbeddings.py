# pre compute the embeddings for the words....
# consider the set of all reviews
from gensim.models import Word2Vec

class MySentences():
    def __init__(self,fname):
        self.fname = fname

    def __iter__(self):
        for file in self.fname:
            with open(file+'_filtered') as fs:
                for line in fs:
                    label = line[0]
                    review = line[2:]
                    for sent in review.strip().split('.'):
                        if len(sent)>0:
                            yield [w for w in sent.split()]



if __name__=="__main__":
    fname = ['../../../amazonUser/User_level_train.csv','../../../amazonUser/User_level_validation.csv']
    sentences = MySentences(fname)
    model = Word2Vec(sentences,iter=15,size=200,workers=10) # keep min_count to default value of 5, dimension of vector =
    #  200
    model.save('amazonWord2Vec')
