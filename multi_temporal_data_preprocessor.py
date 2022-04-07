# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:59:48 2022

@author: James
"""

import json 
import random
import time
from PIL import Image
import pandas as pd
from PIL.TiffTags import TAGS
import numpy as np
import statistics

#opening the JSON files that contain the RGB and NDVI information for each field measured during each month
with open('C://Users//James//OneDrive//University//4th year//FYP//temporal//english_region//April//april_stats_RGB.json') as json_file:
    april_data_RGB = json.load(json_file)
with open('C://Users//James//OneDrive//University//4th year//FYP//temporal//english_region//May//may_stats_RGB.json') as json_file:
    may_data_RGB = json.load(json_file)
with open('C://Users//James//OneDrive//University//4th year//FYP//temporal//english_region//June//june_stats_RGB.json') as json_file:
    june_data_RGB = json.load(json_file)
with open('C://Users//James//OneDrive//University//4th year//FYP//temporal//english_region//July//july_stats_RGB.json') as json_file:
    july_data_RGB = json.load(json_file)
with open('C://Users//James//OneDrive//University//4th year//FYP//temporal//english_region//August//august_stats_RGB.json') as json_file:
    august_data_RGB = json.load(json_file)
    
with open('C://Users//James//OneDrive//University//4th year//FYP//temporal//english_region//April//april_stats_NDVI.json') as json_file:
    april_data_NDVI = json.load(json_file)
with open('C://Users//James//OneDrive//University//4th year//FYP//temporal//english_region//May//may_stats_NDVI.json') as json_file:
    may_data_NDVI = json.load(json_file)
with open('C://Users//James//OneDrive//University//4th year//FYP//temporal//english_region//June//june_stats_NDVI.json') as json_file:
    june_data_NDVI = json.load(json_file)
with open('C://Users//James//OneDrive//University//4th year//FYP//temporal//english_region//July//july_stats_NDVI.json') as json_file:
    july_data_NDVI = json.load(json_file)
with open('C://Users//James//OneDrive//University//4th year//FYP//temporal//english_region//August//august_stats_NDVI.json') as json_file:
    august_data_NDVI = json.load(json_file)

    
#all of the crop types investigated in this project
crop_types = ["Field beans", "Grass", "Oilseed rape", "Other crops", "Peas", "Potatoes", "Spring barley", "Spring wheat", "Winter barley", "Winter oats", "Winter wheat"]



red = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]] #set of 2D arrays where the columns represent the 5 months of april to august
green = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]] # rows represent the eleven different crop types
blue = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
NDVI = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
STD_NDVI = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
STD_RED = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
STD_GREEN = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
STD_BLUE = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]


maximum = []
column =0
row = 0
for crop_type in crop_types: #for each crop type
    red[row][0] = april_data_RGB[crop_type]['pixel_means']['R'] #append each measurement of each crop type during each month to the corresponding column and row 
    red[row][1] = may_data_RGB[crop_type]['pixel_means']['R']
    red[row][2] = june_data_RGB[crop_type]['pixel_means']['R']
    red[row][3] = july_data_RGB[crop_type]['pixel_means']['R']
    red[row][4] = august_data_RGB[crop_type]['pixel_means']['R']
    
    green[row][0] = april_data_RGB[crop_type]['pixel_means']['G']
    green[row][1] = may_data_RGB[crop_type]['pixel_means']['G']
    green[row][2] = june_data_RGB[crop_type]['pixel_means']['G']
    green[row][3] = july_data_RGB[crop_type]['pixel_means']['G']
    green[row][4] = august_data_RGB[crop_type]['pixel_means']['G']
    
    blue[row][0] = april_data_RGB[crop_type]['pixel_means']['B']
    blue[row][1] = may_data_RGB[crop_type]['pixel_means']['B']
    blue[row][2] = june_data_RGB[crop_type]['pixel_means']['B']
    blue[row][3] = july_data_RGB[crop_type]['pixel_means']['B']
    blue[row][4] = august_data_RGB[crop_type]['pixel_means']['B']
    
    NDVI[row][0] = april_data_NDVI[crop_type]['pixel_means']['NDVI']
    NDVI[row][1] = may_data_NDVI[crop_type]['pixel_means']['NDVI']
    NDVI[row][2] = june_data_NDVI[crop_type]['pixel_means']['NDVI']
    NDVI[row][3] = july_data_NDVI[crop_type]['pixel_means']['NDVI']
    NDVI[row][4] = august_data_NDVI[crop_type]['pixel_means']['NDVI']
    
    STD_NDVI[row][0] = april_data_NDVI[crop_type]['pixel_std']['NDVI']
    STD_NDVI[row][1] = may_data_NDVI[crop_type]['pixel_std']['NDVI']
    STD_NDVI[row][2] = june_data_NDVI[crop_type]['pixel_std']['NDVI']
    STD_NDVI[row][3] = july_data_NDVI[crop_type]['pixel_std']['NDVI']
    STD_NDVI[row][4] = august_data_NDVI[crop_type]['pixel_std']['NDVI']
    
    STD_RED[row][0] = april_data_RGB[crop_type]['pixel_std']['R']
    STD_RED[row][1] = may_data_RGB[crop_type]['pixel_std']['R']
    STD_RED[row][2] = june_data_RGB[crop_type]['pixel_std']['R']
    STD_RED[row][3] = july_data_RGB[crop_type]['pixel_std']['R']
    STD_RED[row][4] = august_data_RGB[crop_type]['pixel_std']['R']
    
    STD_GREEN[row][0] = april_data_RGB[crop_type]['pixel_std']['G']
    STD_GREEN[row][1] = may_data_RGB[crop_type]['pixel_std']['G']
    STD_GREEN[row][2] = june_data_RGB[crop_type]['pixel_std']['G']
    STD_GREEN[row][3] = july_data_RGB[crop_type]['pixel_std']['G']
    STD_GREEN[row][4] = august_data_RGB[crop_type]['pixel_std']['G']
    
    STD_BLUE[row][0] = april_data_RGB[crop_type]['pixel_std']['B']
    STD_BLUE[row][1] = may_data_RGB[crop_type]['pixel_std']['B']
    STD_BLUE[row][2] = june_data_RGB[crop_type]['pixel_std']['B']
    STD_BLUE[row][3] = july_data_RGB[crop_type]['pixel_std']['B']
    STD_BLUE[row][4] = august_data_RGB[crop_type]['pixel_std']['B']
    row+=1

min_samples = []
for crop in red: #for each crop type
    flag=1
    min_no_of_samples=0
    for month in crop: #for each set of measurements taken in each month for each crop type
        if flag == 1:
            min_no_of_samples = len(month)
            flag =0
        elif len(month)<min_no_of_samples: #if the number of fields measured during this month was lower than any before
            min_no_of_samples=len(month) #lowest number of measurements updated
    min_samples.append(min_no_of_samples) #list of the minimum number of data samples for each crop type across all months


data_samples = np.zeros((sum(min_samples), 41)) #an array with as rows columns as there will be datasamples and 41 columns representing each bands measurement on each month
row=0
for crop in range(0,11): #for each crop type
    no_of_samples = min_samples[crop] #smallest number of data sampels gathered for the given crop type
    for sample in range(0,no_of_samples): #for every data sample of the given crop type
        data_samples[row][0]=crop #the first column takes the ground truth label value
        for month in range(1,6):
            data_samples[row][month]=red[crop][month-1][sample] #the 41 columns in the array are filled for each data sample with their corresponding values
            data_samples[row][month+5]=green[crop][month-1][sample]
            data_samples[row][month+10]=blue[crop][month-1][sample]
            data_samples[row][month+15]=NDVI[crop][month-1][sample]
            data_samples[row][month+20]=STD_RED[crop][month-1][sample]
            data_samples[row][month+25]=STD_GREEN[crop][month-1][sample]
            data_samples[row][month+30]=STD_BLUE[crop][month-1][sample]
            data_samples[row][month+35]=STD_NDVI[crop][month-1][sample]
        row+=1

col_names = ["target", "red_april", "red_may", "red_june", "red_july", "red_august", "green_april", "green_may", "green_june", "green_july", "green_august", "blue_april", "blue_may", "blue_june", "blue_july", "blue_august"
             , "NIR_april", "NIR_may", "NIR_june", "NIR_july", "NIR_august", "red_std_april", "red_std_may", "red_std_june", "red_std_july", "red_std_august", "green_std_april", "green_std_may", "green_std_june", "green_std_july", "green_std_august"
             , "blue_std_april", "blue_std_may", "blue_std_june", "blue_std_july", "blue_std_august", "NIR_std_april", "NIR_std_may", "NIR_std_june", "NIR_std_july", "NIR_std_august"]
    
data_samples_list = data_samples.tolist()
dataframe = pd.DataFrame(data_samples_list)
df = dataframe
df.columns = col_names


df = df[df.red_april != 254] #simple filter removes any sample in which R, G or B is 254 as this only happens when there are clouds in the data sample.
df = df[df.red_may != 254]
df = df[df.red_june != 254]
df = df[df.red_july != 254]
df = df[df.red_august != 254]



#df.to_csv("C://Users//James//OneDrive//University//4th year//FYP//temporal//english_region//datasets_with_clouds//english_region_cloudy.csv", index =False, header = False) #dataset saved to csv file 
