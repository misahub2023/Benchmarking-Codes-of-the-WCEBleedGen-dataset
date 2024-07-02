

import tensorflow as tf
from model_segment import unet,segnet,linknet
from tensorflow.keras.optimizers import Adam


def load_model_architecture(model_name, input_shape, filters=None):
    if model_name == "unet":
        if filters is None:
            raise ValueError("Filters must be provided for UNet model")
        return unet(size=input_shape[0], num_filters=filters)
    elif model_name == "segnet":
        return segnet(input_shape)
    elif model_name == "linknet":
        return linknet(input_shape)
    else:
        raise ValueError(f"Unsupported model name '{model_name}'")

def compile_model(model, learning_rate=0.001):
    # Compile the model with binary crossentropy loss and metrics
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model
