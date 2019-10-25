from torchvision import models
from torch import nn


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()

        model = models.resnet50(pretrained=True)

        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Conv2d(2048, 1000, kernel_size=3, padding=1)

    def forward_once(self, x):

        y = self.feature_extractor(x)
        y = self.fc(y)

        return y.view(x.shape[0], -1)

    def forward(self, img, img2):

        x1 = self.feature_extractor(img)
        x2 = self.feature_extractor(img2)
        x1 = self.fc(x1)
        x2 = self.fc(x2)

        return x1.view(img.shape[0], -1), x2.view(img.shape[0], -1)


class Simple(nn.Module):
    def __init__(self, kernel_size=5):

        super(Simple, self).__init__()

        self.conv1 = nn.Conv2d(3, 20, kernel_size=kernel_size)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.nolinear1 = nn.ReLU()

        self.conv2 = nn.Conv2d(20, 50, kernel_size=kernel_size)
        self.pool2 = nn.MaxPool2d(2, 2)


    def forward(self, x1, x2):

        out1 = self.nolinear1(self.pool1((self.conv1(x1))))
        out2 = self.nolinear1(self.pool1((self.conv1(x2))))
        out1 = self.pool2((self.conv2(out1)))
        out2 = self.pool2((self.conv2(out2)))
        out1 = out1.view(x1.size(0), -1)
        out2 = out2.view(x2.size(0), -1)

        return out1, out2
