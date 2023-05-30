from torch import nn
import torch
from torchvision.models import resnet34


class XVectors(nn.Module):
    def __init__(
        self,
        input_size: int = 70,
        output_size: int = 12,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, 512, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.rl1 = nn.ReLU()
        self.conv2 = nn.Conv1d(512, 512, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.rl2 = nn.ReLU()
        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, stride=3)
        self.bn3 = nn.BatchNorm1d(512)
        self.rl3 = nn.ReLU()
        self.conv4 = nn.Conv1d(512, 512, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.rl4 = nn.ReLU()
        self.conv5 = nn.Conv1d(512, 1500, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm1d(1500)
        self.rl5 = nn.ReLU()
        self.segment6 = nn.Linear(3000, 512)
        self.sgm6 = nn.Sigmoid()
        self.segment7 = nn.Linear(512, 512)
        self.sgm7 = nn.Sigmoid()
        self.output = nn.Linear(512, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.rl1(self.bn1(self.conv1(x)))
        x = self.rl2(self.bn2(self.conv2(x)))
        x = self.rl3(self.bn3(self.conv3(x)))
        x = self.rl4(self.bn4(self.conv4(x)))
        x = self.rl5(self.bn5(self.conv5(x)))
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        stat_pooling = torch.cat((mean, std), 1)
        x = self.sgm6(self.segment6(stat_pooling))
        x = self.sgm7(self.segment7(x))
        x = self.softmax(self.output(x))
        return x


if __name__ == "__main__":
    _ = XVectors()
