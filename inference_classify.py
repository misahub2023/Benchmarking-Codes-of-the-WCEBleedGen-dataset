import argparse
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from model_classify import create_model, get_optimizer

def load_images_with_datagen(folder_path, datagen):
    image_paths = []
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = datagen.standardize(image)  # Standardize the image
            images.append(image)
            image_paths.append(image_path)
    return np.array(images), image_paths

def main(args):
    datagen = ImageDataGenerator(rescale=1./255)

    x_test, image_paths = load_images_with_datagen(args.test_dir, datagen)
    
    model = create_model(args.base_model)
    model.load_weights(args.model_weights)
    
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=args.loss,
        metrics=args.metrics
    )
    
    predictions = model.predict(x_test)
    predictions=np.argmax(predictions,axis=1)
    predictions=['Bleeding' if i==1 else 'Non-Bleeding' for i in predictions]
    for i, pred in enumerate(predictions):
        print(f"Image: {image_paths[i]}")
        print(f"Predictions: {pred}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a trained model")
    parser.add_argument('--test_dir', type=str, required=True, help="Directory containing test images")
    parser.add_argument('--base_model', type=str, required=True, help="Base model to use (e.g., VGG19, ResNet50V2)")
    parser.add_argument('--model_weights', type=str, required=True, help="Path to the model weights file (.h5)")
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer to use", choices=['Adam', 'SGD', 'RMSprop'])
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument('--loss', type=str, default='categorical_crossentropy', help="Loss function to use")
    parser.add_argument('--metrics', type=str, nargs='+', default=['accuracy'], help="Metrics for evaluation")
    
    args = parser.parse_args()
    main(args)
