import os
import shutil
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import numpy as np
import json

# create folders for training and testing in the same location as where our dataset is saved
config = json.load(open("./TrainingConfig.json", "r"))
datasetPath = config["LPD"].get("PATH")
trainDataPath = os.path.join(datasetPath, "trainData")
testDataPath = os.path.join(datasetPath, "testData")
os.mkdir(trainDataPath)
os.mkdir(testDataPath)

# loop through Dataset to split and format into training and testing folders
inTraining = True
for dirname, _, filenames in os.walk(os.path.join(datasetPath, "annotations")):
    for filename in filenames:
        # get image filename by convert from xml to png (image has same name as annotation)
        imagename = filename.replace("xml", "png")
        # split dataset between training and testing
        if inTraining:
            # copy image and annotation into trainData folder
            shutil.copy(os.path.join(dirname, filename), os.path.join(trainDataPath, filename))
            shutil.copy(os.path.join(os.path.join(datasetPath, "images"), imagename), os.path.join(trainDataPath, imagename))
            inTraining = False
        else:
            # copy image and annotation into testData folder
            shutil.copy(os.path.join(dirname, filename), os.path.join(testDataPath, filename))
            shutil.copy(os.path.join(os.path.join(datasetPath, "images"), imagename), os.path.join(testDataPath, imagename))
            inTraining = True
        print(os.path.join(dirname, filename))

# preprocess data and train model
train = core.Dataset(trainDataPath,transform=transforms.Compose([transforms.ToPILImage(),transforms.ColorJitter(saturation=0.2),transforms.ToTensor(),utils.normalize_transform(),]))
test = core.Dataset(testDataPath)
dataLoader = core.DataLoader(train, shuffle=True)
model = core.Model(['licence'])
loss = model.fit(dataLoader, test, epochs=2, lr_step_size=5, learning_rate=0.005, verbose=True)

# save model
model.save('./Models/LPD.pth')