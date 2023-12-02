import os
import shutil
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import numpy as np
import json

flag = True
config = json.load(open("./config.json", "r"))
datasetPath = config["LPDDAC"]["PATH"]
trainDataPath = os.path.join(datasetPath, "/trainData")
testDataPath = os.path.join(datasetPath, "/testData")
os.mkdir(trainDataPath)
os.mkdir(testDataPath)

# Extract and format training data from the dataset
for dirname, _, filenames in os.walk(os.path.join(datasetPath, "/annotations")):
    for filename in filenames:
        imagename = filename.replace("xml", "png")
        if flag:
            shutil.copy(os.path.join(dirname, filename),
                        trainDataPath+filename)
            shutil.copy(os.path.join(
                os.path.join(datasetPath, "/images"), imagename), trainDataPath+imagename)
            flag = False
        else:
            shutil.copy(os.path.join(dirname, filename),
                        testDataPath+filename)
            shutil.copy(os.path.join(
                os.path.join(datasetPath, "/images"), imagename), testDataPath+imagename)
            flag = True
        print(os.path.join(dirname, filename))

# Preprocess the data and train the model
train = core.Dataset(trainDataPath, transform=transforms.Compose([transforms.ToPILImage(
), transforms.ColorJitter(saturation=0.2), transforms.ToTensor(), utils.normalize_transform(),]))
test = core.Dataset(testDataPath)
dataLoader = core.DataLoader(train, shuffle=True)
model = core.Model(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                   'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
loss = model.fit(dataLoader, test, epochs=10, lr_step_size=5,
                 learning_rate=0.005, verbose=True)

# Save the model
model.save('./Models/LPDDAC.pth')
