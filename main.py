import os
from VehicleDetection.execute import handler as VD_handler
from LicensePlateDetection.execute import handler as LPD_handler
from LicensePlateDigitDetectionAndClassification.execute import handler as LPDDAC_handler
from detecto.visualize import show_labeled_image
from PIL import Image


if __name__ == "__main__":
    try:
        for img in os.listdir('./Images'):
            img_path = './Images/' + img

            print('processing image: ', img)
            # Call VD_handler to process the image and save path in img variable
            img1, boxes1, labels1 = VD_handler(img_path)
            # crop Image according to annotation (boxes)
            with Image.open(img_path) as image:
                # Iterate over each set of bounding box dimensions in the tensor
                for i, (xmin, ymin, xmax, ymax) in enumerate(boxes1):
                    # Crop the image
                    cropped_image = image.crop((xmin.item(), ymin.item(), xmax.item(), ymax.item()))
                    # Save the cropped image
                    cropped_image.save(f"./CroppedImages/cropped_image_{i}.png")
            
                    # Call LPD_handler to process the image and save path in img variable
                    img2, boxes2, labels2 = LPD_handler(f"./CroppedImages/cropped_image_{i}.png")

                    with Image.open(f"./CroppedImages/cropped_image_{i}.png") as image2:           
                        cropped_image2 = image2.crop((boxes2[0][0].item(), boxes2[0][1].item(), boxes2[0][2].item(), boxes2[0][3].item()))
                        # Save the cropped image
                        cropped_image2.save(f"./CroppedImages/cropped_image_{i}.png")
                        
                        # Call LPDDAC_handler to process the image
                        img3, boxes3, labels3 = LPDDAC_handler(f"./CroppedImages/cropped_image_{i}.png") 

                        # output the resulting image {Have to change this to send the image to the server}
                        print('image processed!')
                        show_labeled_image(img3, boxes3, labels3)
       

    except KeyboardInterrupt:
        print('\n\n\ninterrupted!\n\n\n')
