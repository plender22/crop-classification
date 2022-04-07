import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import pandas as pd
import os 
from torch.utils.data import Dataset
from skimage import io


#function that allows for data samples to be stored in a directory and their filenames and targest to be contained in a csv file
class custom_dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0]) #getting all of the filenames of each data samples
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1])) #gets all of the targets i.e. ground truth labels from csv file
        
        if self.transform:
            image = self.transform(image) #transforming image into pytorch tensor
            
        return(image, y_label) # image and its groun truth label returned


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # will run on gpu if there is one

in_channels =3 #each chanel corresponds to a different spectral band. In this case the three are RGB values
input_size = 65536 #no. of pixels in each image presented to the CNN
num_classes = 11 #number of crop types being investigated
learning_rate = 1e-3 #determines step size in gradient descent
batch_size = 16 #number of samples in each batch
num_epochs = 1 # number of times the model is trained on the same image

train_dataset = custom_dataset(csv_file = "/home/jovyan/work/analysis/split_then_augment/augmented_train_labels.csv", root_dir = '/home/jovyan/work/analysis/split_then_augment/augmented_train_images/', 
                         transform = transforms.ToTensor()) # csv file containing file names of the images and ground truth labels and the root directory were the images are stored.

test_dataset = custom_dataset(csv_file = "/home/jovyan/work/analysis/split_then_augment/test_labels.csv", root_dir = '/home/jovyan/work/analysis/split_then_augment/test_images/', 
                         transform = transforms.ToTensor())


train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)  #True for shuffle means that for each epoch, the batches are shuffled
test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True)

model = torchvision.models.resnet18(pretrained=True) #defining the CNN architecture to be used, pretraining means that it is initially trained on googlenet
model.to(device)

criterion = nn.CrossEntropyLoss() #loss function
optimizer = optim.Adam(model.parameters(),lr=learning_rate) #gradient descent algorithm


for epoch in range(num_epochs): #how many times we train on each image 
    for batch_inx, (data, targets) in enumerate(train_loader): #batch_inx allows us to see what batch index we are at
        data = data.to(device = device)
        targets = targets.to(device=device)
        
        scores = model(data) #forward pass through the network
        loss = criterion(scores, targets) # the value from the loss function
        
        
        optimizer.zero_grad() #reset the gradients after each batch
        loss.backward() #backward propagation
        
        optimizer.step() #gradient decent calculated from backward propagation
        train_loss=loss.item()


def check_recall(loader,model):
    classifications = []
    ground_truth = []
    num_correct = 0
    num_samples = 0
    model.eval() # set the model into evaluation mode 
    
    with torch.no_grad(): # tells pytorch to not calculate the gradients for the following
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x) #getting the outputted classifications for each inputted image
            scores.max()
            _, predictions = scores.max(1) #"_," holds the value from the last execution #predictions holds the models classifications of each image
            predict_temp = (predictions.cpu()).tolist() # returning the tensor with the results from the gpu to cpu
            y_temp = (y.cpu()).tolist()
            indx = 0
            for i in predict_temp:
                classifications.append(str(i)) #getting all of the classifications made by the model
                ground_truth.append(str(y_temp[indx])) #getting all of the corresponding ground truth labels
                indx = indx + 1
                
            num_correct += (predictions == y).sum() #number of predicitons that match the correct label
            num_samples += predictions.size(0) #number of samples used
        print(f'got {num_correct} / {num_samples} with recall {float(num_correct)/float(num_samples)*100:.2f}')
        return(classifications, ground_truth)
        

model.eval() #model into evaluation mode
y_pred, y_true = check_recall(test_loader, model) #will print the achieved overall recall of the classifier