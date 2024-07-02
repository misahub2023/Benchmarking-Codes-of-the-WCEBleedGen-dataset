import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from segment_model_loader import load_model_architecture, compile_model

def read_image(path):
    x = cv2.imread(path)
    if x is None:
        raise FileNotFoundError(f"Image not found at {path}")
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224, 224))
    x = x / 255.0
    return x


def display_segmentation(image_path, model_name, weights_path, input_shape, filters=None):
  
    model = load_model_architecture(model_name, input_shape, filters)


    model = compile_model(model)

    model.load_weights(weights_path)

    x = read_image(image_path)

    y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
    y_pred = cv2.resize(y_pred.astype(np.uint8), (x.shape[1], x.shape[0]))
    overlay = x.copy()
    overlay[y_pred == 1] = [0, 255, 0]  

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(x)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Segmentation Overlay")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Inference Script")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_name", type=str, required=True, choices=["unet", "segnet", "linknet"], help="Model architecture to use")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the model weights file")
    parser.add_argument("--input_shape", type=int, nargs='+', default=[224, 224, 3], help="Input shape for the model (default: 224 224 3)")
    parser.add_argument("--filters", type=int, nargs='+', default=[64, 128, 256, 512], help="Filters for UNet model (default: [64, 128, 256, 512])")
    args = parser.parse_args()

    display_segmentation(args.image_path, args.model_name, args.weights_path, args.input_shape, args.filters)