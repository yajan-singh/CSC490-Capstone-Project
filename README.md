# CSC409-Capstone-Project
Licence plate detection

Group: Pixel Pioneers

Requirements Installation: To set up the environment for this project, please install Detecto by running the following command in your terminal:
pip3 install detecto

Running the Code:
To execute the main program, start “CSC490-Capstone-Project/main.py” which will take any images in ./Image and it will one by one extract and output an annotated image of the license plate and all its digits.

While the code is running it will go through each model and the outputted cropped image of each model will be stored in ./CroppedImages

To train models on your own dataset: To do this you may edit the PATH in the TrainingConfig.json file to lead to whatever dataset you choose and you can run train.py in the folder of whatever model you wish to train.
