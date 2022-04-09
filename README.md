# crop-classification
<p align="center">
  <img width="460" height="350" src="https://i.ibb.co/SRtp1wX/Screenshot-2022-04-07-093342.png">
</p>
This project aimed to build a machine learning model proficient at crop type classification. Such a model would be able to accurately discern what crop type was planted
within a field when presented with an image of that field as taken by a satellite. The three main models investigated were: a mono-temporal convolutional neural network, a
multi-temporal artificial neural network and a multi-temporal random forest. 

The code used both run all three models as well as that used to build datasets applicable for their training and testing is contained within this repository. The dataset
pre-processing algorithms take in the raw data inputs of Sentinel 2 imagery and Digimap crop maps and return data samples of differing configurations depending
for what model the dataset is being built for. 

Within this repository there are seven algorithms with the following functions:
1.	cloud_calculator.py calculates the percentage of cloud cover in Sentinel 2 imagery.
2.	CNN_dataset_creator.py takes Sentinel 2 imagery as an input and outputs cropped and blotted images of individual fields applicable for the training and testing of CNNs. All images are 256x256 pixels.
3.	CNN_model.py is the convolutional neural network model that was used in this project.
4.	multi_temporal_data_processor.py takes the inputs of Sentinel 2 RGB and NDVI measurements and outputs the average and standard deviations of these bands for each field.
5.	multi_temporal_dataset_creator.py takes the data samples created by multi_temporal_data_process.py and splits them into training and testing subsets with the option of augmenting the training set being available.
6.	ANN_model.py is the artificial neural network model used in this project.
7.	RF_model.py is the random forest model used in this project. 
