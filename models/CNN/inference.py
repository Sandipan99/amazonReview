# inference module for char level cnn
import torch
import numpy as np
import csv
from torch.utils.data import DataLoader
from torch.autograd import Variable
from char_cnn_2 import *
#torch.cuda.set_device('cpu')

# modify to include the reviewer id
class InfDataset(Dataset):
    def __init__(self, data_path, class_path=None, max_length=1014):
        self.data_path = data_path
        self.vocabulary = list(""" abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'/\|_@#$%ˆ&*˜‘+-=<>()[]{}""")
        self.identity_mat = np.identity(len(self.vocabulary))
        texts, labels, reviewerID = [], [], []
        with open(data_path,encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                if idx%1000000==0:
                    print('processed reviews: ',idx)
                ID = line[0]
                text = ""
                for tx in line[1:-1]:
                    text += tx
                    text += " "    
                     
                label = int(line[-1])
                texts.append(text)
                labels.append(label)
                reviewerID.append(ID)
                
        self.texts = texts
        self.labels = labels
        self.reviewerID = reviewerID
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
        revID = self.reviewerID[index]
        return data, label, revID
 


def inference(batch_size):

    test_params = {"batch_size": batch_size,
                   "shuffle": False,
                   "num_workers": 0}

    test_set = InfDataset('User_level_test_with_id.csv')

    print('dataset created')

    # generators for our training and test sets
    test_generator = DataLoader(test_set, **test_params)

    print('dataloader created... beginning testing')

    model = torch.load("trained_model")
    model.eval()
    with torch.no_grad():

        test_true = []
        test_prob = []
        test_id = []

        batch_count = 0

        for batch in test_generator:
            batch_count+=1
            if batch_count%100==0:
                print('processed batch: ',batch_count)
            _, n_true_label, id_ = batch

            #batch = [Variable(record).cuda() for record in batch]

            t_data, _, _ = batch
            t_data = t_data.cuda()
            t_predicted_label = model(t_data)

            test_prob.append(t_predicted_label)
            test_true.extend(n_true_label)
            test_id.extend(id_)
            

        test_prob = torch.cat(test_prob, 0)
        test_prob = test_prob.cpu().data.numpy()
        test_true = np.array(test_true)
        test_pred = np.argmax(test_prob, -1)

        #print(test_pred)

    # fieldnames = ['True label', 'Predicted label', 'Text']
    # with open(datapath + "output_reddit.csv", 'w',encoding='utf-8') as csv_file:
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
    #     writer.writeheader()
    #     for i, j, k in zip(test_true, test_pred, test_set.texts):
    #         writer.writerow(
    #             {'True label': i, 'Predicted label': j, 'Text': k})

    #test_metrics = get_evaluation(test_true, test_prob,
    #                              list_metrics=["accuracy", "loss", "confusion_matrix"])
    #print("Prediction:\nLoss: {} Accuracy: {} \nConfusion matrix: \n{}".format(test_metrics["loss"],
    #                                                                           test_metrics["accuracy"],
    #                                                                           test_metrics["confusion_matrix"]))
    fieldnames = ['TrueLabel', 'PredictedLabel', 'ReviewerID']
    with open('output_Char_cnn_2.csv','w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(test_true, test_pred, test_id):
           writer.writerow( {'TrueLabel': i, 'PredictedLabel': j, 'ReviewerID': k})



if __name__ == "__main__":
    inference(1024)

