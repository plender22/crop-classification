# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:41:53 2022

@author: James
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


train_dataset = pd.read_csv("C://Users//James//OneDrive//University//4th year//FYP//temporal//train//train_dataset_augmented.csv", header = None) #training dataset
y_train = train_dataset.iloc[:,0].values
x_train = train_dataset.iloc[:,1:41].values

test_dataset = pd.read_csv("C://Users//James//OneDrive//University//4th year//FYP//temporal//2020_validation_region//2020_validation_dataset.csv", header = None)#validation dataset
y_test = test_dataset.iloc[:,0].values
x_test = test_dataset.iloc[:,1:41].values


scaler = StandardScaler() #selecting feature scaling algorithm
X_train = scaler.fit_transform(x_train) #applying feature scaling
X_test = scaler.transform(x_test)

classifier = RandomForestClassifier(n_estimators = 150, criterion = 'entropy', random_state = 42)#defining RF model
classifier.fit(X_train, y_train) # training the RF model

y_pred = classifier.predict(X_test) # Predicting the Test set results



#--- code used for testing below ----


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

