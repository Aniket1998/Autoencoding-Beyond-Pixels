import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Encoder', 'Decoder', 'Discriminator']


class Encoder(nn.Module):
    def __init__(self, nonlinearity=None):
        nl = nn.ReLU() if nonlinearity is not None else nonlinearity
        self.conv = nn.Sequential(nn.Conv2d(3, 64, 5,stride = 2,padding = 2),
                                  nn.BatchNorm2d(64), nl,
                                  nn.Conv2d(64, 128, 5,stride = 2,padding = 2),
                                  nn.BatchNorm2d(128), nl,
                                  nn.Conv2d(128, 256, 5,stride = 2,padding = 2),
                                  nn.BatchNorm2d(256), nl)
        self.mean_fc = nn.Sequential(nn.Linear(8 * 8 * 256, 2048),
                                     nn.BatchNorm1d(2048), nl)
        self.logvar_fc = nn.Sequential(nn.Linear(8 * 8 * 256, 2048),
                                       nn.BatchNorm1d(2048), nl)

    def forward(self, x):
        x = self.conv(x).view(-1, x.size(1) * x.size(2) * x.size(3))
        mu = self.mean_fc(x)
        logvar = self.logvar_fc(x)
        z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar) if self.training else mu
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, nonlinearity=None):
        nl = nn.ReLU() if nonlinearity is not None else nonlinearity
        self.fc = nn.Sequential(nn.Linear(2048, 8 * 8 * 256),
                                nn.BatchNorm1d(8 * 8 * 256), nl)

        self.conv = nn.Sequential(nn.ConvTranspose2d(256, 256, 5,stride = 2,padding = 2,output_padding = 1),
                                  nn.BatchNorm2d(256), nl,
                                  nn.ConvTranspose2d(256, 128, 5,stride = 2,padding = 2,output_padding = 1),
                                  nn.BatchNorm2d(128), nl,
                                  nn.ConvTranspose2d(128, 32, 5,stride = 2,padding = 2,output_padding = 1),
                                  nn.BatchNorm2d(32), nl,
                                  nn.ConvTranspose2d(32, 3, 5,stride = 2,padding = 2,output_padding = 1), nn.Tanh())

    def forward(self, x):
        x = self.fc(x).view(-1, 256, 8, 8)
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, nonlinearity=None):
        nl = nn.ReLU() if nonlinearity is not None else nonlinearity

        self.main = nn.Sequential(nn.Conv2d(3, 32, 5), nl,
                                  nn.Conv2d(32, 128, 5),
                                  nn.BatchNorm2d(128), nl,
                                  nn.Conv2d(128, 256, 5),
                                  nn.BatchNorm2d(256), nl,
                                  nn.Conv2d(256, 256, 5), nl,
                                  nn.Linear(, 1), nn.Sigmoid())

    def forward(self, x):
        return self.main(x)
