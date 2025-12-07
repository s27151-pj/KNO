# main.py
import argparse
from train_engine import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fashion-MNIST training")

    parser.add_argument("--model-type", choices=["dense", "cnn"], default="dense")
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path or f"artifacts/fashion_{args.model_type}.keras"
    metrics_path = args.metrics_path or f"artifacts/metrics_{args.model_type}.json"

    train_model(
        model_type=args.model_type,
        use_augmentation=args.use_augmentation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=model_path,
        metrics_path=metrics_path,
    )


if __name__ == "__main__":
    main()
