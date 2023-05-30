from src.data.common_voice_datamodule import CommonVoiceDataModule
from torch.utils.data import DataLoader, Dataset

dm = CommonVoiceDataModule()
dm.setup()
dl = dm.train_dataloader()
print(dl)
for i in dl:
    print(i)
    break
