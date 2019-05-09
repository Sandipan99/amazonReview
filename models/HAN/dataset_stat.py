# calculate the average length of reviews in terms of number of sentences....
# calculate the average length of sentences....

import pickle
import numpy as np
from Heirarchicalnet import creatingDataset, text2tensor

with open('word2index.pickle','rb') as fs:
    w2i = pickle.load(fs)

train_dataset = creatingDataset('../../../amazonUser/User_level_train.csv',w2i) 

lengths = list(train_dataset.keys())
lengths.sort()

sent_lengths = []
dist_length = {}

for l in lengths:
    dist_length[l] = len(train_dataset[l])
    for rev in train_dataset[l]:
            s = [len(s) for s in rev[0]]
            sent_lengths+=s


#with open('rev_length_dist.pickle','wb') as ft:
#    pickle.dump(dist_length,ft)

for i in range(1,len(lengths)):
    l_c = lengths[i]
    l_p = lengths[i-1]
    dist_length[l_c]+=dist_length[l_p]

max_length = lengths[-1] 

sum_ = dist_length[max_length]

print('max_length: ',max_length)

for l in lengths:
    dist_length[l]/=sum_
    if dist_length[l]>0.95:
        print('length containing 95% of mass: ',l)
        break

print('mean length of sentences: ',np.mean(sent_lengths))
print('std length of sentences: ',np.std(sent_lengths))


