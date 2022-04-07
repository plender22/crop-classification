import torch
import numpy
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

class NN(nn.Module): #neural network model
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__() #super calls the parent initialisation class from nn.Module
        self.fc1 = nn.Linear(input_size, 50) #first hidden layer of the neural network
        self.fc2 = nn.Linear(50, num_classes) #second hidden layer of neural network
        
    def forward(self, x): #forward propagation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Dataset(Dataset):
    def __init__(self, csv_file): 
        scaler = StandardScaler() #feature scaling algorithm 
        xy = np.loadtxt(csv_file, delimiter=",", dtype = np.float32) #dataset to numpy array
        self.x = torch.from_numpy(scaler.fit_transform(xy[:,1:])) #all data within datasamples to tensor
        self.y = torch.from_numpy(xy[:, [0]]) # all ground truth labels from datasamples to tensor
        self.n_samples = xy.shape[0] #number of samples in dataset
        
    def __len__(self):
            return(self.n_samples)
        
    def __getitem__(self, index):
        return(self.x[index], int((self.y[index])))
        
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # will run on gpu if there is one

input_size = 40 #no. of inputs
num_classes = 11 #number of crop types
learning_rate = 0.001 #leaning rate defining step size in gradient descent
batch_size =16 #number of data samples in each batch
num_epochs = 15 # number of times the model is trained on the same image


train_set = Dataset(csv_file = "train_dataset_augmented.csv") #training dataset 

test_set = Dataset(csv_file = "test_dataset.csv") #validation dataset

train_loader = DataLoader(dataset=train_set, batch_size = batch_size, shuffle=True)  #True for shuffle means that for each epoch, the batches are shuffled
test_loader = DataLoader(dataset = test_set, batch_size=batch_size, shuffle=True)


model = NN(input_size = input_size, num_classes=num_classes).to(device) #defining that the NN model is to be used

criterion = nn.CrossEntropyLoss() #loss function
optimizer = optim.Adam(model.parameters(),lr=learning_rate) #gradient decent

for epoch in range(num_epochs): #how many times we train on each image 
    for batch_inx, (data, targets) in enumerate(train_loader): #batch_inx allows us to see what batch index we are at
        data = data.to(device = device) # all tensor data to gpu
        targets = targets.to(device=device) #getting classification from data
     
        scores = model(data) #forward pass through the network
        loss = criterion(scores, targets) # the value from the loss function
        
        optimizer.zero_grad() #reset the gradients after each batch
        loss.backward() #backward propagation
        
        optimizer.step() #gradient decent calculated from backward propagation
        
def check_accuracy(loader,model):
    classifications = []
    ground_truth = []
    num_correct = 0
    num_samples = 0
    model.eval() # set the model into evaluation mode 
    
    with torch.no_grad(): # tells pytorch to not calculate the gradients for the following calculations 
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            scores.max()
            _, predictions = scores.max(1) #"_," holds the value from the last execution #predictions holds the models classifications of each image
            predict_temp = (predictions.cpu()).tolist()
            y_temp = (y.cpu()).tolist()
            indx = 0
            for i in predict_temp:
                classifications.append(str(i))
                ground_truth.append(str(y_temp[indx]))
                indx = indx + 1
                
            num_correct += (predictions == y).sum() #number of predicitons that match the correct label
            num_samples += predictions.size(0) #number of samples used
        print(f'got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        return(classifications, ground_truth)
        

model.eval()
y_pred, y_true = check_accuracy(test_loader, model)
model.train()

y_pred = np.asarray(y_pred)
y_test = np.asarray(y_true)


true_positives = [0,0,0,0,0,0,0,0,0,0,0]
false_negatives = [0,0,0,0,0,0,0,0,0,0,0]
true_negatives = [0,0,0,0,0,0,0,0,0,0,0]
false_positives = [0,0,0,0,0,0,0,0,0,0,0]
crop_types = ["Field beans", "Grass", "Oilseed rape", "Other crops", "Peas", "Potatoes", "Spring barley", "Spring wheat", "Winter barley", "Winter oats", "Winter wheat"]

for i in range(0, len(y_test)):
    if y_pred[i] == y_test[i]: #if classification correct
        true_positives[int(y_test[i])]+=1 
        index = 0
        for i in true_negatives:
            if index != y_pred[i]:
                true_negatives[index]+=1
            index+=1
            
    else: # if classification is incorrect
        false_negatives[int(y_test[i])]+=1
        false_positives[int(y_pred[i])]+=1
        index = 0
        for i in true_negatives:
            if index != y_pred[i] and index != y_test[i]:
                true_negatives[index]+=1
            index+=1
        
recalls = [0,0,0,0,0,0,0,0,0,0,0]
precisions = [0,0,0,0,0,0,0,0,0,0,0]
accuracies = [0,0,0,0,0,0,0,0,0,0,0]

for i in range(0, len(true_positives)): #calculating precision and recall for each crop type
    if true_positives[i]+false_negatives[i] != 0: 
        recalls[i] = (true_positives[i]/(true_positives[i]+false_negatives[i]))
    else:
        recalls[i] = 0
        
    if true_positives[i]+false_positives[i] != 0:
        precisions[i] = (true_positives[i]/(true_positives[i]+false_positives[i]))
    else:
        precisions[i] = 0
    accuracies[i] = ((true_positives[i]+true_negatives[i])/(true_positives[i]+false_positives[i]+true_negatives[i]+false_negatives[i]))
    
for i in range(0, len(true_positives)): #printing individual crop type precision and recall
    print(crop_types[i], ", Accuracy: ",accuracies[i], ", Recall ", recalls[i])
    
#Printing overall classifier metrics
total_precision = (sum(true_positives)/(sum(true_positives)+sum(false_positives)))
total_recall = (sum(true_positives)/(sum(true_positives)+sum(false_negatives)))
print("Overall Accuracy: ", ((sum(true_positives)+sum(true_negatives))/(sum(true_positives)+sum(true_negatives)+sum(false_positives)+sum(false_negatives))))
print("Overall Precision: ", total_precision)
print("Overall Recall: ", total_recall)
print("Overall F1: ", 2*((total_recall*total_precision)/(total_recall+total_precision)))

