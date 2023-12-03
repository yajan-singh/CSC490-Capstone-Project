import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# ML
import os
import shutil
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms

from os import listdir
from os.path import isfile, join
from PIL import Image 

import os
import shutil
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import numpy as np
import json 

config = json.load(open("../TrainingConfig.json", "r"))
datasetPath = config["VD"]["PATH"]
trainDataPath = os.path.join(datasetPath, "trainData")
testDataPath = os.path.join(datasetPath, "testData")

# os.mkdir(trainDataPath)
# os.mkdir(testDataPath)

#This can be used if annotations are in a csv file
# import csv 

# training_files =  [f for f in listdir('/kaggle/input/car-object-detection/data/training_images') if isfile(join('/kaggle/input/car-object-detection/data/training_images', f))]
# def convert_row(row):
#     img = Image.open("/kaggle/input/car-object-detection/data/training_images/" + row[0]) 
#     return """
#    <annotation>
#     <folder>images</folder>
#     <filename>%s</filename>
#     <size>
#         <width>%s</width>
#         <height>%s</height>
#         <depth>3</depth>
#     </size>
#     <object>
#         <name>car</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <occluded>0</occluded>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>%s</xmin>
#             <ymin>%s</ymin>
#             <xmax>%s</xmax>
#             <ymax>%s</ymax>
#         </bndbox>
#     </object>
# </annotation>
# """ % (row[0],img.width, img.height, row[1], row[2], row[3], row[4])

             
# f = open('/kaggle/input/car-object-detection/data/train_solution_bounding_boxes (1).csv')
# csv_f = csv.reader(f)   
# data = []

# for row in csv_f: 
    
#     if row[0] in training_files:
#         f_write = open("./trainData/"+ row[0][:-4] + ".xml", "w")
#         f_write.write(convert_row(row))
#         f_write.close()
#         shutil.copy(os.path.join('/kaggle/input/car-object-detection/data/training_images/', row[0]), "./trainData/" + row[0])
        
# f.close()

flag = True


print(datasetPath)
for dirname, _, filenames in os.walk(os.path.join(datasetPath, "train")):
    for filename in filenames:
        shutil.copy(os.path.join(dirname, filename), os.path.join(trainDataPath, filename))
for dirname, _, filenames in os.walk(os.path.join(datasetPath, "test")):
    for filename in filenames:
        print(os.path.join(testDataPath, filename))
        shutil.copy(os.path.join(dirname, filename), os.path.join(testDataPath, filename))


train = core.Dataset(trainDataPath)
test = core.Dataset(testDataPath)
dataLoader=core.DataLoader(train, shuffle=True)
model = core.Model(['car'])
loss = model.fit(dataLoader, train, epochs=1, lr_step_size=5, learning_rate=0.001, verbose=True)

model.save('../Models/VD.pth')