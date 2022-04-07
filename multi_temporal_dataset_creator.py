# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:55:40 2022

@author: James
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

#original dataset containing all data samples
dataset = pd.read_csv("C://Users//James//OneDrive//University//4th year//FYP//temporal//english_region//english_region_dataset_unaugmented.csv", header = None)

y = dataset.iloc[:,0].values # the ground truth labels
x = dataset.iloc[:,1:41].values # the values of RGB and NDVI std and average for each data sample 
yx = dataset.iloc[:,0:41].values

df_xy = pd.DataFrame(yx) #dataset to dataframe
train, test = train_test_split(df_xy, test_size=0.25, random_state=1) #splitting the dataframe up into 75% train 25% test

#labels for every column in the dataset
col_names = ["target", "red_april", "red_may", "red_june", "red_july", "red_august", "green_april", "green_may", "green_june", "green_july", "green_august", "blue_april", "blue_may", "blue_june", "blue_july", "blue_august"
             , "NIR_april", "NIR_may", "NIR_june", "NIR_july", "NIR_august", "red_std_april", "red_std_may", "red_std_june", "red_std_july", "red_std_august", "green_std_april", "green_std_may", "green_std_june", "green_std_july", "green_std_august"
             , "blue_std_april", "blue_std_may", "blue_std_june", "blue_std_july", "blue_std_august", "NIR_std_april", "NIR_std_may", "NIR_std_june", "NIR_std_july", "NIR_std_august"]
train.columns = col_names #assigning column names to datasets
test.columns = col_names

train.sort_values(by=["target"]) #numerically arranging the split datasets in terms of their ground truth label from 0 to 10
test.sort_values(by=["target"])

max_size = train['target'].value_counts().max() #getting the most prevalent data classes number of samples

lst = pd.DataFrame(columns = col_names) #blank dataframe 

for class_index, group in train.groupby('target'):
    temp = group.sample(max_size-len(group), replace=True) #resample if the given dataclass if not the most populated one
    lst = lst.append(temp, ignore_index=True) #resampled/augmented instances stored in datafram
    
augmented_train = train.append(lst, ignore_index=True) # original training dataset split earlier has augmented samples added, balancing it out

df = augmented_train
#df.to_csv("C://Users//James//OneDrive//University//4th year//FYP//temporal//2020_validation_region//2020_train_dataset_augmented.csv", index =False, header = False) #saving the augmented training dataset


df = test
#df.to_csv("C://Users//James//OneDrive//University//4th year//FYP//temporal//2020_validation_region//2020_test_dataset_unaugmented.csv", index =False, header = False) #saving the unaugmented validation dataset







