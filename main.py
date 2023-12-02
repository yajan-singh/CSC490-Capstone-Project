import os
from LicensePlateDigitDetectionAndClassification.execute import handler as LPDDAC_handler
from detecto.visualize import show_labeled_image

if __name__ == "__main__":
    try:
        while True:
            if len(os.listdir('./Images')) == 0:
                print('waiting for image...')
                continue
            img = os.listdir('./Images')[0]
            print('processing image: ', img)
            # TODO: Call VD_handler to process the image and save path in img variable(Vivek)
            # TODO: Call LPD_handler to process the image and save path in img variable (Nawal)
            # Call LPDDAC_handler to process the image
            img, boxes, labels = LPDDAC_handler('./Images/'+img)
            # output the resulting image {Have to change this to send the image to the server}
            print('image processed!')
            show_labeled_image(img, boxes, labels)

    except KeyboardInterrupt:
        print('\n\n\ninterrupted!\n\n\n')
