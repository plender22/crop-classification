import json 
from PIL import Image
import pandas as pd

location = "20190826-ROI1"

with open('/home/jovyan/work/croprcnn/new_data/preprocessed/RGB/train/via_region_data.json') as json_file:
    data = json.load(json_file) #getting the JSON file returned by Wiktor Jurek's cod 
    
excel_df = pd.read_csv("/home/jovyan/work/analysis/english_region/labels.csv", names = ['name', 'target']) #getting data of any samples currently in the dataset


cropped_file_names = []
targets = []


def crop_type_to_no(crop_type): #simple ground truth label to numerical target conversion
    target = 22
    if crop_type == "Field beans":
        target = 0
    if crop_type == "Grass":
        target = 1
    if crop_type == "Oilseed rape":
        target = 2
    if crop_type == "Other crops":
        target = 3
    if crop_type == "Peas":
        target = 4
    if crop_type == "Potatoes":
        target = 5
    if crop_type == "Spring barley":
        target = 6
    if crop_type == "Spring wheat":
        target = 7
    if crop_type == "Winter barley":
        target = 8
    if crop_type == "Winter oats":
        target = 9
    if crop_type == "Winter wheat":
        target = 10
    return(target)

# please refer to https://dirask.com/posts/JavaScript-how-to-calculate-intersection-point-of-two-lines-for-given-4-points-VjvnAj in order to understand how the following code works
def point_of_intersection_check(x1, y1, x2, y2, x3, y3, x4, y4): #co-ordinates of four points
    y1 = -1*y1 #pixel co-ordinates start form 0 at the top of the image and increase as you go "down" in the image
    y2 = -1*y2
    y3 = -1*y3
    y4 = -1*y4
    
    denominator = (((x1-x2)*(y3-y4))-((y1-y2)*(x3-x4))) #denominator of the intersection of two lines from four points that lie on that line equation
    if denominator == 0:
        return(False)
    else:
        Px = ((((x1*y2)-(y1*x2))*(x3-x4))-((x1-x2)*((x3*y4)-(y3*x4))))/denominator #determining the x-coordinate of intersection of two lines
        Py = ((((x1*y2)-(y1*x2))*(y3-y4))-((y1-y2)*((x3*y4)-(y3*x4))))/denominator #determining the y-coordinate of intersectin of two lines
        if ((Py>=y1 and Py<=y2) or (Py>=y2 and Py<=y1)) and ((Px>=x1 and Px<=x2) or (Px>=x2 and Px<=x1)) and (Px>=x3): #checking if the point of interception is within the boundaries of the field
            return(True)
        else:
            return(False)


def pixel_inside_check(scaled_x_points, scaled_y_points, pixel_x, pixel_y, max_x_point): #checks if a line drawn from each pixel to the right intersects with aboundary of the field
    counter =0
    for i in range(0,(len(x_points)-1)): #for each edge of the fields polygone shape
        x1 = scaled_x_points[i] #x co-ordinate of the first point on the edge
        y1 = scaled_y_points[i] #y co-ordinate of the first point on the edge
        x2 = scaled_x_points[i+1] #x co-ordinate of the second point on the edge
        y2 = scaled_y_points[i+1] #y co-ordinate of the second point on the edge
        intersection_check = point_of_intersection_check(x1,y1,x2,y2,pixel_x,pixel_y,pixel_x+1,pixel_y)
        if intersection_check == True: #if a line drawn form the pixel investigated to the right of the image intersects a field boundary then true
            counter = counter + 1 #counts how many boundaries the line drawn from the pixel intersects
    
    if counter%2 == 0: #a pixel within the field will always intersect the boundaries of that field an uneven number of times
        return("outside")
    else:
        return("inside")

        
    
        
    

x=0
for i in data: #for every patch
    filename = data[i]['filename'] #getting the name of the patch
    for d in data[i]['regions']:# for each individual field in the patch
        img = Image.open('/home/jovyan/work/croprcnn/new_data/preprocessed/RGB/train/'+filename) #getting the 2D data from the image
        
        pixels = img.load() #allows for each pixel in the image to be manipulated
        x_points = data[i]['regions'][d]['shape_attributes']['all_points_x'] #get the x co-ordinates of the fields polygone shape
        y_points = data[i]['regions'][d]['shape_attributes']['all_points_y'] #get the y co-ordinates of the fields polygone shape
        crop_type = data[i]['regions'][d]['region_attributes']['name'] #get the ground truth label of what crop is planted with the field
        max_x_point = max(x_points) #getting the extremes of the verticies
        max_y_point = max(y_points)
        min_x_point = min(x_points)
        min_y_point = min(y_points)
        distance_x = max_x_point - min_x_point #getting the dimentions of the bounding box the field will be cropped with
        distance_y = max_y_point - min_y_point
        
        if distance_x <= distance_y: #if the bounding box is taller than it is wide
            img_cropped = img.crop((int(min_x_point), int(min_y_point), int(min_x_point)+int(distance_x), int(min_y_point) + int(distance_x))) #crop individual field out of landscape using bounding box
            
        if distance_y < distance_x: #if the bounding wider than it is tall
            img_cropped = img.crop((int(min_x_point), int(min_y_point), int(min_x_point)+int(distance_y), int(min_y_point) + int(distance_y)))
            
        x_origin_change = min_x_point #finding what the co-ordinates of the top left pixel in the newly cropped image will be 
        y_origin_change = min_y_point
        
        #changing the verticies of the fields to allign with the cropped version of the image
        scaled_x_points = []
        for x_point in x_points:
            scaled_x_points.append(x_point-x_origin_change)
        
        scaled_y_points = []
        for y_point in y_points:
            scaled_y_points.append(y_point-y_origin_change)
        
        pixels = img_cropped.load()
        for x in range(0, img_cropped.size[0]):
            for y in range(0, img_cropped.size[0]):
                inside_test = pixel_inside_check(scaled_x_points, scaled_y_points, x, y, img_cropped.size[0]) #checking if every pixel in the cropped image comes from within the field
                if inside_test == "outside": #if the pixel investigated comes from outwidth the field then it is set to be black or "blotted out"
                    pixels[x,y] = (0,0,0)
        
            
        
            
            
        
        newsize = (256,256) # resizing every image so that it is 256x256 pixels for use with the CNN
        img_resized = img_cropped.resize(newsize)

        name= filename[0:len(filename)-4] + '_'+ location +'_region_'+ d + '_processed.jpg'
        cropped_file_names.append(name) #adding name of individually cropped field to the dataset
        target = crop_type_to_no(crop_type) #adding the ground truth of the cropped field to the dataset
        targets.append(target)
        
        destination = '/home/jovyan/work/analysis/english_region/blotted_images/' + name
        img_resized.save(destination, "JPEG", quality=80, optimize=True, progressive=True) #saving the individually cropped field to the dataset directory

d = {'name':cropped_file_names, 'target':targets}
df = pd.DataFrame(d)
new_excel_df = excel_df.append(df, ignore_index=True)
new_excel_df.to_csv("/home/jovyan/work/analysis/english_region/labels.csv", index =False, header = False)#adding the filenames and ground truth labels of the newly cropped fields to the dataset csv file