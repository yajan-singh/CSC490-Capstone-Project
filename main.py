import os
from VehicleDetection.execute import handler as VD_handler
from LicensePlateDetection.execute import handler as LPD_handler
from LicensePlateDigitDetectionAndClassification.execute import handler as LPDDAC_handler
from detecto.visualize import show_labeled_image
from PIL import Image

def crop_image(image_path, crop_dimensions):
    """
    Crops an image to multiple bounding boxes and saves each crop as a new file.

    :param image_path: Path to the source image.
    :param crop_dimensions: List of bounding box coordinates in the format [xmin, ymin, xmax, ymax].
    """
    # Load the image
    with Image.open(image_path) as img:
        # Iterate over each set of bounding box dimensions in the tensor
        for i, (xmin, ymin, xmax, ymax) in enumerate(crop_dimensions):
            # Crop the image
            cropped_image = img.crop((xmin.item(), ymin.item(), xmax.item(), ymax.item()))
            # Save the cropped image
            cropped_image.save("./CroppedImages/cropped_image_{i}.png")

if __name__ == "__main__":
    try:
        while True:
            if len(os.listdir('./Images')) == 0:
                print('waiting for image...')
                continue
            img = os.listdir('./Images')[0]
            img_path = './Images/' + img

            print('processing image: ', img)
            # Call VD_handler to process the image and save path in img variable
            img, boxes, labels = VD_handler(img_path)
            
            # crop Image according to annotation (boxes)
            crop_image(img_path, boxes)

            # Call LPD_handler to process the image and save path in img variable
            for i in range(len(boxes)):
                img, boxes, labels = LPD_handler("./CroppedImages/cropped_image_{i}.png")
                crop_image("./CroppedImages/cropped_image_{i}.png", boxes)
            
                # Call LPDDAC_handler to process the image
                img, boxes, labels = LPDDAC_handler()
                # output the resulting image {Have to change this to send the image to the server}
                print('image processed!')
                show_labeled_image(img, boxes, labels)

    except KeyboardInterrupt:
        print('\n\n\ninterrupted!\n\n\n')
