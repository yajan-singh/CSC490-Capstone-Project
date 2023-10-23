import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Reshape

# Define the model architecture
def create_car_detection_model(input_shape):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    x = base_model.output
    x = Conv2D(4, (3, 3), activation='relu')(x)  # Output layer with 4 channels (for bounding box coordinates)
    x = Reshape((4,))(x)  # Flatten the output to (x, y, width, height)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    
    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

# Define the input shape (adjust according to your dataset)
input_shape = (224, 224, 3)  # Change the dimensions as per your data

# Create the model
model = create_car_detection_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display the model summary
model.summary()
