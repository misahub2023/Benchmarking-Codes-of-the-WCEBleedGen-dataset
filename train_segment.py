import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from model_segment import unet, segnet, linknet
from data_loader_segment import load_data, tf_dataset
from metrics_segment import iou, dice_coefficient

def main():
    tf.config.run_functions_eagerly(True)
    parser = argparse.ArgumentParser(description="Train Segmentation Model")
    parser.add_argument("--model", type=str, choices=["unet", "segnet", "linknet"], required=True, help="Choose the model type: unet, segnet, linknet")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size (default: 224)")
    parser.add_argument("--filters", type=int, nargs="+", default=[64, 128, 256, 512], help="Filters for UNet (default: [64, 128, 256, 512])")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer (default: 0.0001)")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs (default: 250)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--validation_size", type=float, default=0.2, help="Validation size (default: 0.2)")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test size (default: 0.1)")
    args = parser.parse_args()

    # Load data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(args.data_path, validation_size=args.validation_size, test_size=args.test_size)

    # Create TensorFlow datasets
    train_data = tf_dataset(train_x, train_y, batch_size=args.batch_size)
    val_data = tf_dataset(valid_x, valid_y, batch_size=args.batch_size)
    test_data = tf_dataset(test_x, test_y, batch_size=args.batch_size)

    # Define input shape
    input_shape = (args.input_size, args.input_size, 3)

    # Build model
    if args.model == "unet":
        model = unet(args.input_size, args.filters)
    elif args.model == "segnet":
        model = segnet(input_shape)
    elif args.model == "linknet":
        model = linknet(input_shape)

    # Compile model
    opt = Adam(learning_rate=args.learning_rate)
    metrics = ["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou, dice_coefficient]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

    # CSV Logger for training history
    csv_logger = CSVLogger("model_history_log.csv", append=True)

    # Determine steps per epoch for training and validation
    train_steps = len(train_x) // args.batch_size
    valid_steps = len(valid_x) // args.batch_size
    if len(train_x) % args.batch_size != 0:
        train_steps += 1
    if len(valid_x) % args.batch_size != 0:
        valid_steps += 1

    # Training the model
    model.fit(train_data,
              validation_data=val_data,
              epochs=args.epochs,
              steps_per_epoch=train_steps,
              validation_steps=valid_steps,
              callbacks=[csv_logger])

    # Save model weights
    model.save_weights(f"{args.model}.weights.h5")

    # Evaluate on validation dataset
    val_metrics = model.evaluate(val_data, steps=valid_steps)
    print("\nValidation Metrics:")
    print(f"Loss: {val_metrics[0]}")
    print(f"Accuracy: {val_metrics[1]}")
    print(f"Recall: {val_metrics[2]}")
    print(f"Precision: {val_metrics[3]}")
    print(f"IoU: {val_metrics[4]}")
    print(f"Dice Coefficient: {val_metrics[5]}")

    # Evaluate on test dataset
    test_steps = len(test_x) // args.batch_size
    if len(test_x) % args.batch_size != 0:
        test_steps += 1
    test_metrics = model.evaluate(test_data, steps=test_steps)
    print("\nTest Metrics:")
    print(f"Loss: {test_metrics[0]}")
    print(f"Accuracy: {test_metrics[1]}")
    print(f"Recall: {test_metrics[2]}")
    print(f"Precision: {test_metrics[3]}")
    print(f"IoU: {test_metrics[4]}")
    print(f"Dice Coefficient: {test_metrics[5]}")

if __name__ == "__main__":
    main()
