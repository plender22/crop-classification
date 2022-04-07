from os import listdir
from os.path import isfile, join
import rasterio as rio



mypath = "/home/jovyan/work/croprcnn/data/raw/North_Dundee/20200817/ROI1/S2/Patches" #path where all sentinel2 patches are
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] #get the filename of each patch in the path

#print(onlyfiles)
band =16 #cloud mask band

counter1 =0
for patch in onlyfiles: #for each patch
    src = rio.open(mypath+"/"+patch) #open each patch
    cloud_mask = src.read(band) #get cloud mask of each patch
    for i in cloud_mask: #for each row of pixels
        for ii in i: #for each pixel in the row 
            if ii == 0: #if the pixel is designated as being covered by cloud
                counter1+=1 #increase the counter of number of pixels covered by cloud
                
print((counter1/(len(onlyfiles)*(256*256)))*100) # percentage cloud cover is equal to all of the cloud covered pixels divided by the no of pixels