from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import random
import numpy as np
import torch
import os
import pickle


class CommonVoiceDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/transformed_cv/",
        train_val_test_split: Tuple[int, int, int] = (240_000, 12_000, 24_000),
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 12

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """

    def _dataset_loader(self, path):
        sample = np.load(path)
        max_length = 500
        length = sample.shape[2]
        diff = length - max_length
        if diff > 0:
            left_trim = diff // 2
            right_trim = diff - left_trim
            return torch.Tensor(sample[:, :, left_trim:length-right_trim])
        else:
            left_padding = (500 - length) // 2
            right_padding = (500 - length) - left_padding
            return torch.Tensor(np.pad(sample, ((0, 0), (0, 0), (left_padding, right_padding)), mode='constant'))

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            extensions = ['.npy']
            dataset = DatasetFolder(os.path.join(self.hparams.data_dir[:-1]+'_pl/', 'data'), self._dataset_loader, extensions)

            # with open(os.path.join(self.hparams.data_dir, 'subsets/train_indices.pkl'), 'rb') as file:
            #     train_indices = pickle.load(file)
            # with open(os.path.join(self.hparams.data_dir, 'subsets/val_indices.pkl'), 'rb') as file:
            #     val_indices = pickle.load(file)
            # with open(os.path.join(self.hparams.data_dir, 'subsets/test_indices.pkl'), 'rb') as file:
            #     test_indices = pickle.load(file)

            # train_dataset = Subset(dataset, train_indices)
            # val_dataset = Subset(dataset, test_indices)
            # test_dataset = Subset(dataset, val_indices)


            train_size = int(0.8 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
            self.data_train = train_dataset
            self.data_val = val_dataset
            self.data_test = test_dataset
    
    def _collate_fn(self, batch):
        batch = sorted(batch, key=lambda x: x[0].shape[2], reverse=True)
        max_length = batch[0][0].shape[2]
        padded_batch = []
        outputs = []
        for sample in batch:
            length = sample[0].shape[2]
            padding_length = max_length - length
            padding = torch.zeros((1, 70, padding_length))
            padded_sample = torch.cat([torch.Tensor(sample[0]), padding], dim=2)
            padded_batch.append(padded_sample.unsqueeze(1))
            outputs.append(torch.Tensor([sample[1]]))
        return (torch.cat(padded_batch, dim=0), torch.cat(outputs, dim=0))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            # collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            # collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            # collate_fn=self._collate_fn
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = CommonVoiceDataModule()