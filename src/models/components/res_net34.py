from torch import nn
from torchvision.models import resnet34


class ResNet34(nn.Module):
    def __init__(
        self,
        output_size: int = 12,
    ):
        super().__init__()

        self.model = resnet34(num_classes=output_size)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    _ = ResNet34()
