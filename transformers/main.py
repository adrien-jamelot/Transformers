from transformers.inference.inference import inference
from transformers.training.train import train
import argparse
import mlflow


def main():
    parser = argparse.ArgumentParser(
        description="Train or Predict with a decoder architecture"
    )
    parser.add_argument("mode", default="inference", choices=["inference", "training"])

    args = parser.parse_args()

    if args.mode == "inference":
        decoder = mlflow.pyfunc.load_model("models:/model/1").get_raw_model()
        inference(decoder)
    elif args.mode == "training":
        train()


if __name__ == "__main__":
    main()
