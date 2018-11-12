from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class AENet(nn.Module):
    def __init__(self, image_size):
        super(AENet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(image_size * image_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, image_size * image_size),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x