from torch import Tensor
from torch.utils.data import Dataset

type customDatasetType = tuple[Tensor, Tensor]


class CustomDataset(Dataset[customDatasetType]):
    def __init__(self, data: Tensor, target: list[Tensor]) -> None:
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx) -> customDatasetType:
        return self.data[idx], self.target[idx]
