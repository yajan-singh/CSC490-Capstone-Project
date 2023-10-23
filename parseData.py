
import sys

import tensorflow as tf


def parse_data():
    # Create Dataset Batches
    print("\nParsing Dataset\n")
    BatchSize = 32
    data = tf.keras.preprocessing.image_dataset_from_directory(
        'dataset', batch_size=BatchSize)

    # Scale data to 0-1
    data = data.map(lambda x, y: (x / 255.0, y))
    return data


def split_data(data):
    # Split data into train and test
    train_data = data.take(int(len(data)/2))
    val_data = data.skip(int(len(data)/2)).take(2)
    test_data = data.skip(int(len(data)/2)+2)

    if len(train_data) + len(val_data) + len(test_data) != len(data):
        print("\n!!!!!!!!!!!!!!!! Error in splitting data !!!!!!!!!!!!!!!!\n".upper())
        sys.exit(1)
    return train_data, val_data, test_data
