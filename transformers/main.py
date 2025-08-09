from transformers.inference import inference
from transformers.training.train import train
from transformers.inference.inference import inference
from transformers.blocks.decoder import Decoder
import torch
import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Train or Predict with a decoder architecture"
    )
    parser.add_argument("mode", default="inference", choices=["inference", "training"])

    args = parser.parse_args()

    if args.mode == "inference":
        # with open('model_parameters', 'r') as modelParametersFile:
        #     modelParameters: dict = json.load(modelParametersFile)
        # decoder = Decoder(modelParameters)
        # checkpoint = torch.load("model_epoch_4.pt")
        # decoder.load_state_dict(checkpoint['model_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        # print(f"Loaded checkpoint at epoch {epoch} with loss {loss}")
        # inference(
        #     decoder,
        #     modelParameters,

        # )
        pass
    elif args.mode == "training":
        train()


if __name__ == "__main__":
    main()
