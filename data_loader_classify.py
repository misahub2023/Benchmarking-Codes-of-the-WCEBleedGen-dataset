import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import pickle

def get_data(data_dir):
    X = []
    Y = []

    for category in os.listdir(data_dir):
        path = os.path.join(data_dir, category, 'Images')
        for images in os.listdir(path):
            imagePath = os.path.join(path, images)
            image = cv2.imread(imagePath)

            # Ensure the image is in RGB format (OpenCV reads in BGR format)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            X.append(np.array(image))
            if category == 'bleeding':
                Y.append(1.0)
            else:
                Y.append(0.0)
    return np.array(X), np.array(Y)

def prepare_data(data_dir, test_size=0.30, val_size=0.33, augment=False):
    x, y = get_data(data_dir)

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y, 2)

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=test_size, random_state=379)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=val_size, random_state=379)

    if augment:
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            rescale=1./255
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)
    
    datagen.fit(x_train)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), datagen

def save_data(x_train, y_train, x_val, y_val, x_test, y_test, datagen, output_dir='data'):
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez_compressed(os.path.join(output_dir, 'train_data.npz'), x_train=x_train, y_train=y_train)
    np.savez_compressed(os.path.join(output_dir, 'val_data.npz'), x_val=x_val, y_val=y_val)
    np.savez_compressed(os.path.join(output_dir, 'test_data.npz'), x_test=x_test, y_test=y_test)
    
    with open(os.path.join(output_dir, 'datagen.pkl'), 'wb') as f:
        pickle.dump(datagen, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and preprocess data")
    parser.add_argument('data_dir', type=str, help="Path to the dataset")
    parser.add_argument('--test_size', type=float, default=0.30, help="Test set size ratio")
    parser.add_argument('--val_size', type=float, default=0.33, help="Validation set size ratio from test set")
    parser.add_argument('--augment', action='store_true', help="Apply data augmentation")
    parser.add_argument('--output_dir', type=str, default='data', help="Output directory to save the preprocessed data")

    args = parser.parse_args()

    (x_train, y_train), (x_val, y_val), (x_test, y_test), datagen = prepare_data(
        args.data_dir, args.test_size, args.val_size, args.augment
    )

    print(f"Train data shape: {x_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {x_val.shape}, {y_val.shape}")
    print(f"Test data shape: {x_test.shape}, {y_test.shape}")

    save_data(x_train, y_train, x_val, y_val, x_test, y_test, datagen, args.output_dir)
