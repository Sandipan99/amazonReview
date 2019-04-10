#my training module
#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn import metrics
import sys
from torch.utils.data import Dataset
csv.field_size_limit(sys.maxsize)


#custom dataloader
class MyDataset(Dataset):
    def __init__(self, data_path, class_path=None, max_length=1014):
        self.data_path = data_path
        self.vocabulary = list(""" abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'/\|_@#$%ˆ&*˜‘+-=<>()[]{}""")
        self.identity_mat = np.identity(len(self.vocabulary))
        texts, labels = [], []
        with open(data_path,encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx
                    text += " "
                # if len(line) == 3:
                #     text = "{} {}".format(line[1].lower(), line[2].lower())
                # else:
                #     text = "{}".format(line[1].lower())
                label = int(line[0])
                texts.append(text)
                labels.append(label)
                
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.labels)
        if class_path:
            self.num_classes = sum(1 for _ in open(class_path))

    #gets the length 
    def __len__(self):
        return self.length

    #gets data based on given index
    #done the encoding here itself
    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), len(self.vocabulary)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.vocabulary)), dtype=np.float32)
        label = self.labels[index]
        return data, label
    
    
#char level cnn model
class  CharCNN(nn.Module):
    
    def __init__(self, n_classes, input_dim=69,max_seq_length=1014,filters=256,kernel_sizes=[7,7,3,3,3,3],pool_size=3,
                n_fc_neurons=1024):
        
        super(CharCNN, self).__init__()
        
        self.filters = filters
        self.max_seq_length = max_seq_length
        self.n_classes = n_classes
        self.pool_size = pool_size
        
        #pooling in layer 1,2,6 ; pool =3
        #layers 7,8 and 9 are fully connected
        #2 dropout modules between 3 fully connected layers
        #dropout - prevents overfitting in neural networks ; drops neurons with a certain probability
        #(from 2014 paper Dropout by Srivastava et al.); only during training
        
        #layer 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, filters, kernel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

        #layer 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[1]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )
       
        #layer 3,4,5
        self.conv3 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[2]),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[3]),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[4]),
            nn.ReLU()
        )

        #layer 6
        self.conv6 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[5]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

        dimension = int((max_seq_length - 96) / 27 * filters)
        
        #layer 7
        self.fc1 = nn.Sequential(
            nn.Linear(dimension, n_fc_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        #layer 8
        self.fc2 = nn.Sequential(
            nn.Linear(n_fc_neurons, n_fc_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        #layer 9
        self.fc3 =nn.Linear(n_fc_neurons, n_classes)
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.LogSoftmax()
        
        if filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)
        
        
    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)
        
        
    def forward(self,input):
        
        input = input.transpose(1, 2)
        #print(input.size())
        output = self.conv1(input)
        #print(output.size())
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.sigmoid(output)

        return output
    
#evaluate model
def get_evaluation(y_true, y_prob, list_metrics):
    
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

    
#train model
def train(batch_size,num_epochs,learning_rate):

    training_params = {"batch_size": batch_size,
                           "shuffle": True,
                           "num_workers": 0}
    test_params = {"batch_size": batch_size,
                       "shuffle": False,
                       "num_workers": 0}

    validation_params = {"batch_size": batch_size,
                       "shuffle": False,
                       "num_workers": 0}

    training_set = MyDataset("User_level_train.csv")
    #test_set = MyDataset(datapath + "test.csv")
    validation_set = MyDataset("User_level_validation.csv")

    #generators for our training and test sets
    training_generator = DataLoader(training_set, **training_params)
    #test_generator = DataLoader(test_set,**test_params)
    validation_generator = DataLoader(validation_set,**validation_params)

    #our model
    model = CharCNN(n_classes=2)
    model.cuda()

    #loss function and optimizer
    #using binary cross entropy loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    #train the model; basically telling on what to train
    model.train()

    num_iter_per_epoch = len(training_generator)

    best_accuracy = 0
    output_file = open("logs.txt", "w")
    #with open(datapath + "logs_1.txt", "w") as output_file:
    #training loop
    for epoch in range(num_epochs):

        for iter, batch in enumerate(training_generator):

            #get the inputs
            _, n_true_label = batch

            #wrap them in Variables
            #Variables are specifically tailored to hold values which
            #change during training of a neural network,
            #i.e. the learnable paramaters of our network
            batch = [Variable(record).cuda() for record in batch]

            #final inputs after wrapping
            t_data,t_true_label = batch

            #forward pass: compute predicted y by passing x to the model
            t_predicted_label = model(t_data)
            #print(t_predicted_label[0])
            
            #print(t_predicted_label.size(),t_true_label.size())
            
            #retrieve tensor held by a variable
            n_prob_label = t_predicted_label.cpu().data.numpy()

            #compute loss
            loss = criterion(t_predicted_label, t_true_label)
            #if(iter%1000==0):
                #print("After {} iterations, loss : {}".format(iter+1,loss))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            #compute gradients
            loss.backward()
            optimizer.step() 

            training_metrics = get_evaluation(n_true_label, n_prob_label, list_metrics=["accuracy", "loss"])

            print("Training: Iteration: {}/{} Epoch: {}/{} Loss: {} Accuracy: {}".format(iter + 1, 
                                                                                        num_iter_per_epoch,
                                                                                        epoch + 1, num_epochs,
                                                                                        training_metrics["loss"],
                                                                                        training_metrics[
                                                                                        "accuracy"]))


        #evaluation of validation data
        model.eval()
        with torch.no_grad():

            validation_true = []
            validation_prob = []

            for batch in validation_generator:

                _, n_true_label = batch

                #setting volatile to true because we are in inference mode
                #we will not be backpropagating here
                #conserving our memory by doing this
                #edit:volatile is deprecated now; using torch.no_grad();see above

                batch = [Variable(record).cuda() for record in batch]
                #get inputs
                t_data, _ = batch
                #forward pass
                t_predicted_label = model(t_data)
                #using sigmoid to predict the label
                #t_predicted_label = F.sigmoid(t_predicted_label)

                validation_prob.append(t_predicted_label)
                validation_true.extend(n_true_label)

            validation_prob = torch.cat(validation_prob, 0)
            validation_prob = validation_prob.cpu().data.numpy()
            validation_true = np.array(validation_true)

        
        #back to default:train
        model.train()

        test_metrics = get_evaluation(validation_true, validation_prob,
                                          list_metrics=["accuracy", "loss", "confusion_matrix"])
        
        
        output_file.write(
                "Epoch: {}/{} \nTraining loss: {} Training accuracy: {} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1, num_epochs,
                    training_metrics["loss"],
                    training_metrics["accuracy"],
                    test_metrics["loss"],
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
        print ("\tTest:Epoch: {}/{} Loss: {} Accuracy: {}\r".format(epoch + 1, num_epochs, test_metrics["loss"],
                                                                     test_metrics["accuracy"]))
        
        #acc to the paper; half lr after 3 epochs
        if(num_epochs> 0 and num_epochs%3==0):
            learning_rate = learning_rate/2
            
        #saving the model with best accuracy
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            torch.save(model, "trained_model")


if __name__=='__main__':
    
    #train model and get metrics
    train(1024,10,0.01)
