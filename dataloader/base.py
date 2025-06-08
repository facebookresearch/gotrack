# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict


from pathlib import Path
from torch.utils.data import Dataset
from utils.logging import get_logger

logger = get_logger(__name__)


class GoTrackDataset(Dataset):
    """Dataloader for GoTrack datasets."""

    def __init__(self, root_dir: str, *args, **kwargs) -> None:
        super().__init__()

        # Required parameters.
        self.root_dir = Path(root_dir)
        self.dp_model = None
        self.models = None
        self.models_vertices = None

    def __len__(self):
        raise NotImplementedError("The length of the dataset is not defined.")

    def __getitem__(self, index):
        raise NotImplementedError("The __getitem__ method is not implemented.")
