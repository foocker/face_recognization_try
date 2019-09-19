import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1 , 48->24

            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2 24->12

            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU(),   # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2),   # pool3 12->6

            nn.Conv2d(64,128, kernel_size=2,stride=1),   # conv4 6 -> 4
            nn.PReLU()  # prelu4
        )
        self.conv5 = nn.Linear(128*2*2, 256)  # conv5
        self.prelu5 = nn.PReLU()  # prelu5
        # lanbmark localization
        self.conv6_3 = nn.Linear(256, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        landmark = self.conv6_3(x)
        return landmark