import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.training.chunking import readChunks
import mlflow
import pydantic


def computeNextCharProbas(
    totalString: str,
    contextWindowSize: int,
    char2idx: dict[str, int],
    vocabSize: int,
    decoder: nn.Module,
) -> Tensor:
    inputString = [char2idx[c] for c in totalString[-contextWindowSize:]]
    oneHotInitString = F.one_hot(torch.tensor(inputString).long(), vocabSize).float()
    probs = nn.Softmax(dim=0)(decoder(oneHotInitString.squeeze(0))[0][-1])
    return probs


def computeNextCharLogits(
    totalString: str,
    contextWindowSize: int,
    char2idx: dict[str, int],
    vocabSize: int,
    decoder: nn.Module,
) -> Tensor:
    inputString = [char2idx[c] for c in totalString[-contextWindowSize:]]
    oneHotInitString = F.one_hot(torch.tensor(inputString).long(), vocabSize).float()
    probs = decoder(oneHotInitString.squeeze(0))[0][-1]
    return probs


def beamSearch(
    decoder: nn.Module,
    oneHotInitString: torch.Tensor,
    nbBeams: int,
    nbOutputTokens: int,
    idx2char: list[str],
    char2idx: dict[str, int],
    totalString: str,
    vocabSize: int,
) -> str:
    """This is an attempt at using beam search for decoding instead of
    sampling out of the next character distribution. I wrote a first
    draft but it seems to converge quickly into looping around the
    same word. it may make sense as beam search is still greedy, I
    have tried to provide more flexibility by allowing sampling to
    select the top candidates but this hasn't been successful so
    far. I'll leave it there in case I want to explore more and clean
    that code.

    :param decoder:
    :param oneHotInitString:
    :param nbBeams:
    :param nbOutputTokens:
    :param idx2char:
    :param char2idx:
    :param totalString:
    :param vocabSize:
    :returns:

    """
    probs = nn.Softmax(dim=0)(decoder(oneHotInitString.squeeze(0))[0][-1])
    print(probs)
    values, indices = torch.topk(probs, nbBeams)
    totalStrings = [totalString for _ in range(nbBeams)]
    for step in range(nbOutputTokens):
        candidates = {}
        for i in range(nbBeams):
            totalStrings[i] += idx2char[indices[i]]
            probs = computeNextCharLogits(
                totalStrings[i], 16, char2idx, vocabSize, decoder
            )
            nextValues, nextIndices = torch.topk(probs, nbBeams)
            beamICandidates = {
                totalStrings[i] + idx2char[nextIndices[j]]: values[i] + nextValues[j]
                for j in range(nbBeams)
            }
            candidates = candidates | beamICandidates
        x = torch.tensor(list(candidates.values()))
        softmaxed = F.softmax(x, dim=0)
        candidates = {key: softmaxed[k] for (k, key) in enumerate(candidates)}
        # topKCandidates = sorted(
        #     set(candidates.items()), key=lambda item: item[1], reverse=True
        # )[:nbBeams]
        sample = torch.multinomial(softmaxed, nbBeams, replacement=False)
        items = list(candidates.items())
        topKCandidates = [items[k] for k in sample]
        totalStrings: list[str] = [top[0] for top in topKCandidates]
        values: list[Tensor] = [candidates[totalString] for totalString in totalStrings]

    return totalStrings[values.index(max(values))]


@torch.no_grad()
def inference(decoder, runId):
    tmpPath = pydantic.model_serializer(
        mlflow.artifacts.download_artifacts(run_id=runId, artifact_path="chunks.pkl")
    )
    tokenisationInfo = readChunks(tmpPath.wrapped)
    vocabSize, char2idx, idx2char = (
        tokenisationInfo.vocabSize,
        tokenisationInfo.char2idx,
        tokenisationInfo.idx2char,
    )
    decoder.eval()
    totalString = "ALEXANDRE:\nI do not agree"
    print(f"input:\n{totalString}")
    T = 1
    input = [char2idx[c] for c in totalString[-16:]]
    oneHotInitString = F.one_hot(torch.tensor(input).long(), vocabSize).float()

    for _ in range(500):
        input = [char2idx[c] for c in totalString[-16:]]
        oneHotInitString = F.one_hot(torch.tensor(input).long(), vocabSize).float()

        nextChar = torch.distributions.Categorical(
            probs=nn.Softmax(dim=0)(decoder(oneHotInitString.squeeze(0))[0][-1]) / T
        ).sample()
        totalString = totalString + idx2char[nextChar]

    print("Prediction:")
    # print(
    #     beamSearch(
    #         decoder,
    #         oneHotInitString,
    #         4,
    #         500,
    #         idx2char,
    #         char2idx,
    #         totalString,
    #         vocabSize,
    #     )
    # )
    print(totalString)
    print("---------------------------------------------")
