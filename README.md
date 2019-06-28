# amazonReview

We put all the codes and results related to the paper here.

Added model 1 (1 layer GRU)... it is complete 

Added batch train for GRU...

Modified batch train to prevent memory error

Detecting gender from user name --- use gender-guesser.ipynb


Instructions for running the code...

The preprocessing code <preprocess.py> is in models/HAN
Here you have to specify the location of your train and the validation file... The words will be filtered based on the frequency and the ones with frequency less than 5 will be replaced by <unk> token.
  
The filtered files will be created in the same directory as the original train and validation files
Also a pickle file consisting the word2index map will also be created

Embedding/obtainEmbeddings.py will create the embeddings for the corpus making use of the filtered files

You can directly specify the location of the train and the validation (not the filtered ones), the embedding file (if and when required) and the word2index pickle file in each of the model to train the model.

It will store the model which provides the best validation accuracy
