from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import numpy as np


def load_model():
    model = core.Model.load('./Models/VD.pth', ['car'])
    return model


def handler(img):
    model = load_model()
    img = utils.read_image(img)
    predictions = model.predict(img)
    label, box, num = predictions
    thresh = 0.8
    indices = np.where(num > thresh)
    boxes = box[indices]
    numArr = indices[0].tolist()
    labels = [label[i] for i in numArr]
    return img, boxes, labels