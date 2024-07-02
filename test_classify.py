import argparse
from data_loader_classify import prepare_data
from model_classify import create_model,get_optimizer

def main(args):
    _, _ , (x_test, y_test), datagen = prepare_data(args.data_dir, augment=args.augment)

    model = create_model(args.base_model)
    model.load_weights(args.model_weights)
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=args.loss,
        metrics=args.metrics
    )
    results = model.evaluate(
        datagen.flow(x_test, y_test, shuffle=False),
        verbose=1
    )

    for metric, value in zip(model.metrics_names, results):
        print(f'{metric}: {value:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a trained model")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the preprocessed data")
    parser.add_argument('--base_model', type=str, required=True, help="Base model to use (e.g., VGG19, ResNet50V2)")
    parser.add_argument('--model_weights', type=str, required=True, help="Path to the model weights file (.h5)")
    parser.add_argument('--augment', action='store_true', help="Apply data augmentation if specified")
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer to use", choices=['Adam', 'SGD', 'RMSprop'])
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument('--loss', type=str, default='categorical_crossentropy', help="Loss function to use")
    parser.add_argument('--metrics', type=str, nargs='+', default=['accuracy'], help="Metrics for evaluation")
    args = parser.parse_args()
    main(args)
