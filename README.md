# Benchmarking of the WCEBleedGen Dataset: Classification, Detection, Segmentation
This repository contains the scripts used in the performance evaluation of 16 artificial intelligence models for the [WCEbleedGen Dataset](https://zenodo.org/records/10156571). 10 classification-based, 3 segmentation-based and 3 detection-based pipelines have been trained, validated, and tested. 
The models used are:
* Classification:
   * VGG19
   * Xception
   * ResNet50V2
   * InceptionV3
   * InceptionResNetV2
   * MobileNetV2
   * DenseNet169
   * NasNetMobile
   * EfficientNetB7
   * ConvNeXt
* Segmentation
   * UNet
   * SegNet
   * LinkNet
* Detection
   * YOLOV5nu
   * YOLOV8n
   * YOLOV8x
## Dataset Structure
+ The dataset structure were as follows:
+ datasets/
    + WCEBleedGen/
        + bleeding/
            + images/  
            + bounding_boxes/
                + YOLO-TXT/
            + annotations/
        + non-bleeding/
            + images/
            + annotations/
# Classification
## Loading the data and Preprocessing
* data_loader_classify.py
  
This script is designed to load and preprocess image data for binary classification tasks. It reads images from a directory, processes them, and splits them into training, validation, and test sets. Additionally, it supports data augmentation and saves the preprocessed data for future use.
### Usage

#### Command Line Arguments

- `data_dir`: (Required) Path to the dataset directory.
- `--test_size`: (Optional) Test set size ratio (default is 0.30).
- `--val_size`: (Optional) Validation set size ratio from the test set (default is 0.33).
- `--augment`: (Optional) Apply data augmentation (default is False).
- `--output_dir`: (Optional) Output directory to save the preprocessed data (default is 'data').

#### Example Usage

1. **Basic Usage**

   To preprocess data from a directory `dataset` without data augmentation and save the results to the default output directory:

   ```bash
   python data_loader_classify.py dataset
   ```

2. **With Data Augmentation**

   To apply data augmentation during preprocessing:

   ```bash
   python data_loader_classify.py dataset --augment
   ```

3. **Custom Test and Validation Sizes**

   To set custom test and validation set size ratios:

   ```bash
   python data_loader_classify.py dataset --test_size 0.25 --val_size 0.25
   ```

4. **Custom Output Directory**

   To specify a custom output directory:

   ```bash
   python data_loader_classify.py dataset --output_dir custom_data
   ```

#### Detailed Example

Suppose you have a dataset stored in `my_dataset`, and you want to split the data with a test size of 20% and validation size of 25% from the test set. You also want to apply data augmentation and save the preprocessed data in `processed_data` directory:

```bash
python data_loader_classify.py my_dataset --test_size 0.20 --val_size 0.25 --augment --output_dir processed_data
```

### Output

The script saves the preprocessed data in compressed NumPy format and the data augmentation configuration in a pickle file within the specified output directory. The files generated are:

- `train_data.npz`: Contains training images and labels.
- `val_data.npz`: Contains validation images and labels.
- `test_data.npz`: Contains test images and labels.
- `datagen.pkl`: Contains the data augmentation configuration.
## Defining the models
* model_classify.py
  
This script is designed to create and define deep learning models using various pre-trained architectures available in Keras and TensorFlow. It allows for easy customization of the base model and optimizer, enabling quick experimentation with different configurations.

### Usage

#### Command Line Arguments

- `base_model_name`: (Required) Name of the pre-trained model to use (e.g., `VGG19`, `Xception`, `ResNet50V2`, `InceptionV3`, `InceptionResNetV2`, `MobileNetV2`, `DenseNet169`, `NASNetMobile`, `EfficientNetB7`, `ConvNeXtBase`).
- `optimizer_name`: (Required) Name of the optimizer to use (e.g., `Adam`, `SGD`, `RMSprop`).
- `--learning_rate`: (Optional) Learning rate for the optimizer (default is 0.0001).
- `--input_shape`: (Optional) Input shape of the images (default is (224, 224, 3)).

#### Example Usage

1. **Create a Model with VGG19 and Adam Optimizer**

   To create a model using the VGG19 base model and Adam optimizer with default learning rate:

   ```bash
   python model_classify.py VGG19 Adam
   ```

2. **Custom Learning Rate**

   To use a custom learning rate of 0.001:

   ```bash
   python model_classify.py VGG19 Adam --learning_rate 0.001
   ```

3. **Different Base Model and Optimizer**

   To create a model using the InceptionV3 base model and SGD optimizer:

   ```bash
   python model_classify.py InceptionV3 SGD
   ```

4. **Custom Input Shape**

   To specify a different input shape for the images:

   ```bash
   python model_classify.py ResNet50V2 RMSprop --input_shape 299 299 3
   ```

#### Detailed Example

Suppose you want to create a model using the EfficientNetB7 base model, RMSprop optimizer, with a learning rate of 0.0005, and input shape of 256x256x3:

```bash
python model_classify.py EfficientNetB7 RMSprop --learning_rate 0.0005 --input_shape 256 256 3
```
## Training the models
* train_classify.py

This script is designed to train deep learning models using various pre-trained architectures available in Keras and TensorFlow. It integrates data loading, model creation, and training functionalities, providing an end-to-end solution for model training.

### Usage

#### Command Line Arguments

- `--data_dir`: (Required) Directory containing the preprocessed data.
- `--base_model`: (Required) Base model to use for training (choices are `VGG19`, `Xception`, `ResNet50V2`, `InceptionV3`, `InceptionResNetV2`, `MobileNetV2`, `DenseNet169`, `NASNetMobile`, `EfficientNetB7`, `ConvNeXtBase`).
- `--optimizer`: (Optional) Optimizer to use (default is `Adam`; choices are `Adam`, `SGD`, `RMSprop`).
- `--learning_rate`: (Optional) Learning rate for the optimizer (default is 0.0001).
- `--loss`: (Optional) Loss function to use (default is `categorical_crossentropy`).
- `--metrics`: (Optional) Metrics for evaluation (default is `['accuracy']`).
- `--batch_size`: (Optional) Batch size for training (default is 32).
- `--epochs`: (Optional) Number of epochs to train (default is 10).
- `--model_path`: (Optional) Path to save the trained model (default is `model.h5`).

#### Example Usage

1. **Basic Usage**

   To train a model using the VGG19 base model with default settings and data from the `dataset` directory:

   ```bash
   python train_model.py --data_dir dataset --base_model VGG19
   ```

2. **Custom Optimizer and Learning Rate**

   To use the SGD optimizer with a custom learning rate of 0.001:

   ```bash
   python train_model.py --data_dir dataset --base_model VGG19 --optimizer SGD --learning_rate 0.001
   ```

3. **Different Base Model and Loss Function**

   To train using the InceptionV3 base model with binary cross-entropy loss:

   ```bash
   python train_model.py --data_dir dataset --base_model InceptionV3 --loss binary_crossentropy
   ```

4. **Custom Batch Size and Epochs**

   To specify a batch size of 64 and train for 20 epochs:

   ```bash
   python train_model.py --data_dir dataset --base_model ResNet50V2 --batch_size 64 --epochs 20
   ```

5. **Custom Model Save Path**

   To save the trained model to a custom path:

   ```bash
   python train_model.py --data_dir dataset --base_model EfficientNetB7 --model_path my_model.h5
   ```

#### Detailed Example

Suppose you want to train a model using the EfficientNetB7 base model, RMSprop optimizer, with a learning rate of 0.0005, batch size of 64, for 15 epochs, and save the model to `trained_model.h5`:

```bash
python train_model.py --data_dir dataset --base_model EfficientNetB7 --optimizer RMSprop --learning_rate 0.0005 --batch_size 64 --epochs 15 --model_path trained_model.h5
```

### Output
The model weights are saved at the required location.

## Validation and Testing
* validate_classify.py
* test_classify.py

This guide explains how to use the provided Python scripts to validate and test trained deep learning models. The scripts use various pre-trained architectures available in Keras and TensorFlow. They include functionalities for data loading, model creation, and evaluation.

### Usage

#### 1. **Validation Script: `validate_classify.py`**

This script validates a trained model using the validation dataset.

##### Command Line Arguments

- `--data_dir`: (Required) Directory containing the preprocessed data.
- `--base_model`: (Required) Base model to use for validation (choices: `VGG19`, `Xception`, `ResNet50V2`, `InceptionV3`, `InceptionResNetV2`, `MobileNetV2`, `DenseNet169`, `NASNetMobile`, `EfficientNetB7`, `ConvNeXtBase`).
- `--model_weights`: (Required) Path to the model weights file (.h5).
- `--augment`: (Optional) Apply data augmentation if specified.
- `--optimizer`: (Optional) Optimizer to use (default: `Adam`; choices: `Adam`, `SGD`, `RMSprop`).
- `--learning_rate`: (Optional) Learning rate for the optimizer (default: 0.0001).
- `--loss`: (Optional) Loss function to use (default: `categorical_crossentropy`).
- `--metrics`: (Optional) Metrics for evaluation (default: `['accuracy']`).

##### Example Usage

To validate a model using the VGG19 base model with default settings and data from the `dataset` directory:

```bash
python validate_classify.py --data_dir dataset --base_model VGG19 --model_weights model.h5
```

To use the SGD optimizer with a custom learning rate of 0.001:

```bash
python validate_classify.py --data_dir dataset --base_model VGG19 --model_weights model.h5 --optimizer SGD --learning_rate 0.001
```

#### 2. **Testing Script: `test_classify.py`**

This script tests a trained model using the test dataset.

##### Command Line Arguments

- `--data_dir`: (Required) Directory containing the preprocessed data.
- `--base_model`: (Required) Base model to use for testing (choices: `VGG19`, `Xception`, `ResNet50V2`, `InceptionV3`, `InceptionResNetV2`, `MobileNetV2`, `DenseNet169`, `NASNetMobile`, `EfficientNetB7`, `ConvNeXtBase`).
- `--model_weights`: (Required) Path to the model weights file (.h5).
- `--augment`: (Optional) Apply data augmentation if specified.
- `--optimizer`: (Optional) Optimizer to use (default: `Adam`; choices: `Adam`, `SGD`, `RMSprop`).
- `--learning_rate`: (Optional) Learning rate for the optimizer (default: 0.0001).
- `--loss`: (Optional) Loss function to use (default: `categorical_crossentropy`).
- `--metrics`: (Optional) Metrics for evaluation (default: `['accuracy']`).

##### Example Usage

To test a model using the InceptionV3 base model with default settings and data from the `dataset` directory:

```bash
python test_classify.py --data_dir dataset --base_model InceptionV3 --model_weights model.h5
```

To test using the ResNet50V2 base model with binary cross-entropy loss:

```bash
python test_classify.py --data_dir dataset --base_model ResNet50V2 --model_weights model.h5 --loss binary_crossentropy
```
## Inferencing
* inference_classify.py
This guide explains how to use the `inference_classify.py` script for running inference using a trained deep learning model on test images. The script uses various pre-trained architectures available in Keras and TensorFlow for image classification.

### Usage


#### Command Line Arguments

- `--test_dir`: (Required) Directory containing the test images.
- `--base_model`: (Required) Base model to use for inference (choices: `VGG19`, `Xception`, `ResNet50V2`, `InceptionV3`, `InceptionResNetV2`, `MobileNetV2`, `DenseNet169`, `NASNetMobile`, `EfficientNetB7`, `ConvNeXtBase`).
- `--model_weights`: (Required) Path to the model weights file (.h5).
- `--optimizer`: (Optional) Optimizer to use (default: `Adam`; choices: `Adam`, `SGD`, `RMSprop`).
- `--learning_rate`: (Optional) Learning rate for the optimizer (default: 0.0001).
- `--loss`: (Optional) Loss function to use (default: `categorical_crossentropy`).
- `--metrics`: (Optional) Metrics for evaluation (default: `['accuracy']`).

#### Example Usage

To perform inference using the VGG19 base model with default settings on images from the `test_images` directory:

```bash
python inference_classify.py --test_dir test_images --base_model VGG19 --model_weights model.h5
```

To use the SGD optimizer with a custom learning rate of 0.001:

```bash
python inference_classify.py --test_dir test_images --base_model VGG19 --model_weights model.h5 --optimizer SGD --learning_rate 0.001
```
# Segmentation
## Loading the data- images and masks
This guide explains how to use the data_loader_segment.py script for loading and preparing image segmentation data using TensorFlow and OpenCV. This script loads image data for segmentation tasks, preparing it as TensorFlow datasets.

### Usage

#### Command Line Arguments

- `--path`: (Required) Path to the dataset directory containing subdirectories for each category (`bleeding`, `non-bleeding`) with `Images` and `Annotations` folders.
- `--validation_size`: (Optional) Proportion of validation data (default: 0.2).
- `--test_size`: (Optional) Proportion of test data (default: 0.1).
- `--batch_size`: (Optional) Batch size for training (default: 32).

#### Example Usage

To load data from the `dataset` directory with default settings:

```bash
python data_loader_segment.py --path dataset
```

To customize validation size and batch size:

```bash
python data_loader_segment.py --path dataset --validation_size 0.15 --batch_size 16
```
## Creating and defining the Models
* model_segment.py
  
This script allows the creation and compilation of three different segmentation models: UNet, SegNet, and LinkNet. The script utilizes TensorFlow/Keras for building the models. It also provides an option to specify various hyperparameters such as input size, filters, and learning rate.

### Usage

Run the script from the command line by specifying the model type, input size, filters, and learning rate. Below are examples of how to use the script.

#### Examples

1. **UNet Model**
   ```bash
   python model_segment.py --model unet --input_size 224 --filters 64 128 256 512 --learning_rate 0.001
   ```

2. **SegNet Model**
   ```bash
   python model_segment.py --model segnet --input_size 224 --learning_rate 0.001
   ```

3. **LinkNet Model**
   ```bash
   python model_segment.py --model linknet --input_size 224 --learning_rate 0.001
   ```

### Input Parameters

- `--model`: (Required) Specifies the type of model to create. Choices are "unet", "segnet", "linknet".

- `--input_size`: (Optional) Specifies the size of the input image. Default is 224.

- `--filters`: (Optional) Specifies the number of filters for each convolutional layer in the UNet model. Default is [64, 128, 256, 512]. Only applicable for the UNet model.

- `--learning_rate`: (Optional) Specifies the learning rate for the optimizer. Default is 0.001.
## Training the Models
* metrics_segment.py
* train_segment.py

These scripts are used to train and evaluate the segmentation models.
### Usage
#### metrics_segment.py
This script defines custom metrics and losses used during model training and evaluation.
##### Custom Metrics and Losses
- IoU (Intersection over Union): Measures the overlap between the predicted and true masks.
- Dice Coefficient: Measures the similarity between the predicted and true masks.
- Dice Coefficient Loss: Defined as 1 - Dice Coefficient.
The custom metrics and losses are imported and used in the train_segment.py script for model training and evaluation.
```bash
from metrics_segment import iou, dice_coefficient
```
#### train_segment.py
This script trains and evaluates the segmentation models defined in model_segment.py
Run the script from the command line by specifying the model type, data path, input size, filters, learning rate, number of epochs, batch size, validation size, and test size.

##### Examples

1. **Train UNet Model**
   ```bash
   python train_segment.py --model unet --data_path /path/to/dataset --input_size 224 --filters 64 128 256 512 --learning_rate 0.0001 --epochs 250 --batch_size 32 --validation_size 0.2 --test_size 0.1
   ```

2. **Train SegNet Model**
   ```bash
   python train_segment.py --model segnet --data_path /path/to/dataset --input_size 224 --learning_rate 0.0001 --epochs 250 --batch_size 32 --validation_size 0.2 --test_size 0.1
   ```

3. **Train LinkNet Model**
   ```bash
   python train_segment.py --model linknet --data_path /path/to/dataset --input_size 224 --learning_rate 0.0001 --epochs 250 --batch_size 32 --validation_size 0.2 --test_size 0.1
   ```

#### Input Parameters

- `--model`: (Required) Specifies the type of model to create. Choices are "unet", "segnet", "linknet".
- `--data_path`: (Required) Path to the dataset.
- `--input_size`: (Optional) Specifies the size of the input image. Default is 224.
- `--filters`: (Optional) Specifies the number of filters for each convolutional layer in the UNet model. Default is [64, 128, 256, 512]. Only applicable for the UNet model.
- `--learning_rate`: (Optional) Specifies the learning rate for the optimizer. Default is 0.0001.
- `--epochs`: (Optional) Number of epochs to train the model. Default is 250.
- `--batch_size`: (Optional) Batch size for training. Default is 32.
- `--validation_size`: (Optional) Fraction of the dataset to use for validation. Default is 0.2.
- `--test_size`: (Optional) Fraction of the dataset to use for testing. Default is 0.1.

#### Training and Evaluation

The script loads the dataset, creates TensorFlow datasets for training, validation, and testing, builds the specified model, and trains the model. It also evaluates the model on the validation and test datasets and prints the metrics.

#### Example Output

The script outputs the summary of the created model architecture and compiles the model using the specified learning rate. During training, it saves the training history in a CSV file and saves the model weights. It also prints the validation and test metrics.

## Inferencing
* segment_inference.py

This script is designed for performing inference on images using pre-trained segmentation models. It supports three model architectures: UNet, SegNet, and LinkNet. The script reads an input image, applies the segmentation model, and displays the original image along with the segmentation overlay.

### Usage

Run the script from the command line by specifying the input image path, model architecture, model weights path, input shape, and filters (for UNet only).

#### Example Commands

1. **UNet Model**
   ```bash
   python segment_inference.py --image_path /path/to/image.jpg --model_name unet --weights_path /path/to/weights.h5 --input_shape 224 224 3 --filters 64 128 256 512
   ```

2. **SegNet Model**
   ```bash
   python segment_inference.py --image_path /path/to/image.jpg --model_name segnet --weights_path /path/to/weights.h5 --input_shape 224 224 3
   ```

3. **LinkNet Model**
   ```bash
   python segment_inference.py --image_path /path/to/image.jpg --model_name linknet --weights_path /path/to/weights.h5 --input_shape 224 224 3
   ```

#### Input Parameters

- `--image_path`: (Required) Path to the input image.
- `--model_name`: (Required) Specifies the model architecture to use. Choices are "unet", "segnet", "linknet".
- `--weights_path`: (Required) Path to the model weights file.
- `--input_shape`: (Optional) Input shape for the model. Default is [224, 224, 3].
- `--filters`: (Optional) Filters for the UNet model. Default is [64, 128, 256, 512]. Only applicable for the UNet model.


### Example Output

When the script is run, it displays a window with two images: the original input image and the input image with the segmentation overlay.

```plaintext
Original Image                 Segmentation Overlay
[Shows the input image]        [Shows the input image with green segmentation overlay]
```

This script is useful for visualizing the results of a segmentation model on new images. Modify the script as needed for further customization and experimentation.

# Detection 
For detection, models were used directly from the [ultralytics repositories](https://docs.ultralytics.com/).

# Setup used for Evaluation
All the models were trained for a total of 250 epochs, without any preprocessing or modification. The codes were run using 40 GB DGX A100 NVIDIA GPU workstation available at the Department of Electronics and Communication Engineering, Indira Gandhi Delhi Technical University for Women, New Delhi, India. 

# Results
The results and the findings will be released in the form of a research paper soon, the preprint has been released and can be accessed at [link](https://www.authorea.com/doi/full/10.22541/essoar.171007121.19572474)

# Contributions
Palak Handa conceptualized the research idea, performed the data collection, mask analysis, literature review, and did the research paper writing. Manas Dhir contributed in developing the benchmarking pipeline, developing the github repository, and writing the initial draft of the research paper. Dr. Deepak Gunjan from the Department of Gastroenterology and HNU, AIIMS Delhi performed the medical annotations, and was involved in suggestions for improving artificial intelligence algorithms. Dr. Nidhi Goel was involved in literature review and administration. Jyoti Dhatarwal contributed in the initial data collection. Harshita Mangotra contributed in development of the bounding boxes. Divyansh Nautiyal contributed in correcting the multiple bleeding regions and re-making the bounding boxes, and Nishu, Sanya, Shriya, and Sneha Singh contributed in the result replications on the GPU workstation and table entries. The [WCEbleedGen Dataset](https://zenodo.org/records/10156571) has been actively downloaded more than 1000 times and was utilized in Auto-WCEBleedGen Version 1 and 2 challenge as training dataset. The challenge page is available [here](https://linktr.ee/misahub.challenges).























