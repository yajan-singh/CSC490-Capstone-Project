from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import numpy as np


def load_model():
    model = core.Model.load('./Models/LPDDAC.pth', ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
                                                   'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                                                   'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                                                   'V', 'W', 'X', 'Y', 'Z'])
    return model


def handler(img):
    model = load_model()
    img = utils.read_image(img)
    predictions = model.predict(img)
    label, box, num = predictions
    thresh = 0.50
    indices = np.where(num > thresh)
    boxes = box[indices]
    numArr = indices[0].tolist()
    labels = [label[i] for i in numArr]
    return img, boxes, labels
