from keras.models import Model
from keras.layers import Dense, Flatten
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from tensorflow.keras.applications import ResNet50V2, InceptionV3, InceptionResNetV2, MobileNetV2, DenseNet169, NASNetMobile, EfficientNetB7, ConvNeXtBase
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def create_model(base_model_name, input_shape=(224, 224, 3)):
    base_model = None

    if base_model_name == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'Xception':
        base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'ResNet50V2':
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'InceptionResNetV2':
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'DenseNet169':
        base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'NASNetMobile':
        base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'EfficientNetB7':
        base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'ConvNeXtBase':
        base_model = ConvNeXtBase(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Invalid base model name")

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model

def get_optimizer(optimizer_name, learning_rate=0.0001):
    if optimizer_name == 'Adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        return SGD(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        return RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Invalid optimizer name")
