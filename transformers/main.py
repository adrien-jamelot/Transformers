from transformers.inference.inference import inference
from transformers.training.train import train
import argparse
import mlflow
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(
        description="Train or Predict with a decoder architecture"
    )
    parser.add_argument(
        "mode", choices=["inference", "training"], help="Mode of operation"
    )
    parser.add_argument(
        "--runId", default="5eac5c2b5d624b499176467070add8b9", help="Optional run ID"
    )
    args = parser.parse_args()

    if args.mode == "inference":
        decoder = mlflow.pyfunc.load_model(f"runs:/{args.runId}/decoder")
        decoder = decoder.get_raw_model()
        inference(decoder, args.runId)
    if args.mode == "training":
        train()


if __name__ == "__main__":
    main()
