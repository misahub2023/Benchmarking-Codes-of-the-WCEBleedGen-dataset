import os
import argparse
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import numpy as np

def load_data(path, validation_size=0.2, test_size=0.1):
    categories = ['bleeding', 'non-bleeding']
    data = []

    for category in categories:
        image_folder = os.path.join(path, category, 'Images')
        annotation_folder = os.path.join(path, category, 'Annotations')

        images = sorted(glob(os.path.join(image_folder, "*")))
        annotations = sorted(glob(os.path.join(annotation_folder, "*")))

        category_data = list(zip(images, annotations))

        data.extend(category_data)

    total_size = len(data)
    valid_size = int(validation_size * total_size)
    test_size = int(test_size * total_size)

    train_data, valid_test_data = train_test_split(data, test_size=(valid_size + test_size), random_state=42)
    valid_data, test_data = train_test_split(valid_test_data, test_size=test_size, random_state=42)

    train_x, train_y = zip(*train_data)
    valid_x, valid_y = zip(*valid_data)
    test_x, test_y = zip(*test_data)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224, 224))
    x = x / 255.0
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (224, 224))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])

    x.set_shape([224, 224, 3])
    y.set_shape([224, 224, 1])

    return x, y

def tf_dataset(x, y, batch_size=32):
    x=list(x)
    y=list(y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset

def main(args):
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(args.path, args.validation_size, args.test_size)
    train_dataset = tf_dataset(train_x, train_y, args.batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, args.batch_size)
    test_dataset = tf_dataset(test_x, test_y, args.batch_size)

    return train_dataset, valid_dataset, test_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data loader for segmentation tasks.")
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--validation_size', type=float, default=0.2, help='Proportion of validation data')
    parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of test data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

    args = parser.parse_args()
    train_dataset, valid_dataset, test_dataset = main(args)

    for data in train_dataset.take(1):
        images, masks = data
        print(f'Images batch shape: {images.shape}')
        print(f'Masks batch shape: {masks.shape}')

