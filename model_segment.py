import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dense, Reshape, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define UNet model
def unet(size, num_filters):
    def conv_block(x, num_filters):
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x
    
    inputs = Input((size, size, 3))
    skip_x = []
    x = inputs

    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPooling2D((2, 2))(x)

    x = conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)

# Define SegNet model
def segnet(input_shape):
    img_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same', name='conv1', strides=(1, 1))(img_input)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv6')(x)
    x = BatchNormalization(name='bn6')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv7')(x)
    x = BatchNormalization(name='bn7')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv8')(x)
    x = BatchNormalization(name='bn8')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv9')(x)
    x = BatchNormalization(name='bn9')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv10')(x)
    x = BatchNormalization(name='bn10')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv11')(x)
    x = BatchNormalization(name='bn11')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv12')(x)
    x = BatchNormalization(name='bn12')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv13')(x)
    x = BatchNormalization(name='bn13')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv1')(x)
    x = BatchNormalization(name='bn14')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv2')(x)
    x = BatchNormalization(name='bn15')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv3')(x)
    x = BatchNormalization(name='bn16')(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv4')(x)
    x = BatchNormalization(name='bn17')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv5')(x)
    x = BatchNormalization(name='bn18')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv6')(x)
    x = BatchNormalization(name='bn19')(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv7')(x)
    x = BatchNormalization(name='bn20')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv8')(x)
    x = BatchNormalization(name='bn21')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv9')(x)
    x = BatchNormalization(name='bn22')(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)
    x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv10')(x)
    x = BatchNormalization(name='bn23')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv11')(x)
    x = BatchNormalization(name='bn24')(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)
    x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv12')(x)
    x = BatchNormalization(name='bn25')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(1, (3, 3), padding='same', name='deconv13')(x)
    x = BatchNormalization(name='bn26')(x)
    pred = Activation('sigmoid')(x)
    pred = Reshape((input_shape[0], input_shape[1], 1))(pred)
    return Model(img_input, pred)

# Define LinkNet model
def linknet(input_shape):
    def conv_block(inputs, filters, kernel_size=3, strides=1):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    def encoder_block(inputs, filters, kernel_size=3, strides=1):
        x = conv_block(inputs, filters, kernel_size, strides)
        x = conv_block(x, filters, kernel_size, 1)
        shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(inputs)
        shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    def decoder_block(inputs, filters, kernel_size=3, strides=1):
        x = UpSampling2D(size=(2, 2))(inputs)
        x = conv_block(x, filters, kernel_size, 1)
        x = conv_block(x, filters, kernel_size, 1)
        shortcut = UpSampling2D(size=(2, 2))(inputs)
        shortcut = Conv2D(filters, kernel_size=1, strides=1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    inputs = Input(shape=input_shape)

    # Encoder
    enc1 = encoder_block(inputs, 64, strides=2)
    enc2 = encoder_block(enc1, 128, strides=2)
    enc3 = encoder_block(enc2, 256, strides=2)
    enc4 = encoder_block(enc3, 512, strides=2)

    # Decoder
    dec4 = decoder_block(enc4, 256, strides=2)
    dec3 = decoder_block(dec4, 128, strides=2)
    dec2 = decoder_block(dec3, 64, strides=2)
    dec1 = decoder_block(dec2, 64, strides=2)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(dec1)

    return Model(inputs, outputs)

# Main function to parse arguments and create the model
def main():
    parser = argparse.ArgumentParser(description="Segmentation Model Script")
    parser.add_argument("--model", type=str, choices=["unet", "segnet", "linknet"], required=True, help="Choose the model type: unet, segnet, linknet")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size (default: 224)")
    parser.add_argument("--filters", type=int, nargs="+", default=[64, 128, 256, 512], help="Filters for UNet (default: [64, 128, 256, 512])")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer (default: 0.001)")
    args = parser.parse_args()

    input_shape = (args.input_size, args.input_size, 3)

    if args.model == "unet":
        model = unet(args.input_size, args.filters)
    elif args.model == "segnet":
        model = segnet(input_shape)
    elif args.model == "linknet":
        model = linknet(input_shape)

    model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

if __name__ == "__main__":
    main()


