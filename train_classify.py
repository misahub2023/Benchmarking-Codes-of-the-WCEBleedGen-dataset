import argparse
from model_classify import create_model, get_optimizer
from data_loader_classify import prepare_data

def main(args):
    (x_train, y_train), (x_val, y_val), (x_test, y_test), datagen = prepare_data(args.data_dir)
    model = create_model(args.base_model)
    optimizer = get_optimizer(args.optimizer, args.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=args.loss,
        metrics=args.metrics
    )

    
    model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        validation_data=(x_val, y_val),
        epochs=args.epochs
    )

    model.save(args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the preprocessed data")
    parser.add_argument('--base_model', type=str, required=True, help="Base model to use", choices=['VGG19', 'Xception', 'ResNet50V2', 'InceptionV3', 'InceptionResNetV2', 'MobileNetV2', 'DenseNet169', 'NASNetMobile', 'EfficientNetB7', 'ConvNeXtBase'])
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer to use", choices=['Adam', 'SGD', 'RMSprop'])
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument('--loss', type=str, default='categorical_crossentropy', help="Loss function to use")
    parser.add_argument('--metrics', type=str, nargs='+', default=['accuracy'], help="Metrics for evaluation")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train")
    parser.add_argument('--model_path', type=str, default='model.h5', help="Path to save the trained model")

    args = parser.parse_args()
    main(args)
