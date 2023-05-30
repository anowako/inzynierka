from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import login
from datasets import load_dataset
import numpy as np
import librosa


class CommonVoiceDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (240_000, 12_000, 24_000),
        batch_size: int = 64,
        num_workers: int = 0,
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
        login(token='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')
        load_dataset("mozilla-foundation/common_voice_13_0", "en", streaming=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            cv_13 = load_dataset("mozilla-foundation/common_voice_13_0", "en", streaming=True)
            col_names = cv_13['train'].column_names
            cv_13 = cv_13.filter(self._data_filtering, input_columns=['age', 'gender'])
            cv_13 = cv_13.map(self._data_batch_padding, batched=True, batch_size=self.hparams.batch_size, remove_columns=col_names)
            cv_13 = cv_13.map(self._data_transform)
            self.data_train = cv_13['train']
            self.data_val = cv_13['validation']
            self.data_test = cv_13['test']

    def _data_filtering(self, age, gender):
        age_cond = age in ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties']
        gender_cond = gender in ['male', 'female']
        return age_cond and gender_cond
    
    def _data_batch_padding(self, batch):
        audios = [i['array'] for i in batch['audio']]
        max_len = max([audio.shape[0] for audio in audios])
        output = {'data': list(),
                'label': list(),
                'sr': list()}
        for i, line in enumerate(batch['audio']):
            r = max_len - len(line['array'])
            padded_audio = np.pad(line['array'], (0, r), mode='constant')
            output['data'].append(padded_audio)
            sr = line['sampling_rate']
            output['sr'].append(sr)
            age = batch['age'][i]
            gender = batch['gender'][i]
            age_filter = {'teens': 10,
            'twenties': 20,
            'thirties': 30,
            'fourties': 40,
            'fifties': 50,
            'sixties': 60,
            'seventies': 70,
            'eighties': 80,
            'nineties': 90,}
            label = age_filter[age]/5-2
            if gender == 'male': label += 1
            output['label'].append(label)
        return(output)
    
    def _data_transform(self, batch):
        audio = batch['data']
        sr = batch['sr']
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=256)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=23)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        frame_energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)
        feature_vector = np.vstack((mfcc, mfcc_delta, mfcc_delta2, frame_energy))
        v_a, v_b = feature_vector.shape
        batch['data'] = np.reshape(feature_vector, (1, v_a, v_b))
        return batch

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
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