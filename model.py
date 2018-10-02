import torch
import torch.nn as nn

__all__ = ['Encoder', 'Decoder', 'Discriminator']


class Encoder(nn.Module):
    def __init__(self, nonlinearity=None):
        super(Encoder, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        self.conv = nn.Sequential(nn.Conv2d(3, 64, 5, stride=2, padding=2),
                                  nn.BatchNorm2d(64), nl,
                                  nn.Conv2d(64, 128, 5, stride=2, padding=2),
                                  nn.BatchNorm2d(128), nl,
                                  nn.Conv2d(128, 256, 5, stride=2, padding=2),
                                  nn.BatchNorm2d(256), nl)
        self.mean_fc = nn.Sequential(nn.Linear(8 * 8 * 256, 2048),
                                     nn.BatchNorm1d(2048), nl)
        self.logvar_fc = nn.Sequential(nn.Linear(8 * 8 * 256, 2048),
                                       nn.BatchNorm1d(2048), nl)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        mu = self.mean_fc(x)
        logvar = self.logvar_fc(x)
        z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar) if self.training else mu
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, nonlinearity=None):
        super(Decoder, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        self.fc = nn.Sequential(nn.Linear(2048, 8 * 8 * 256),
                                nn.BatchNorm1d(8 * 8 * 256), nl)

        self.conv = nn.Sequential(nn.ConvTranspose2d(256, 256, 5, stride=2, padding=2, output_padding=1),
                                  nn.BatchNorm2d(256), nl,
                                  nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
                                  nn.BatchNorm2d(128), nl,
                                  nn.ConvTranspose2d(128, 32, 5, stride=2, padding=2, output_padding=1),
                                  nn.BatchNorm2d(32), nl,
                                  nn.ConvTranspose2d(32, 3, 5, stride=1, padding=2), nn.Tanh())

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_normal_(self.conv[len(self.conv) - 2], nn.init.calculate_gain('tanh'))
 

    def forward(self, x):
        x = self.fc(x).view(-1, 256, 8, 8)
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, nonlinearity=None):
        super(Discriminator, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity

        self.conv = nn.Sequential(nn.Conv2d(3, 32, 5, stride=2, padding=2), nl,
                                  nn.Conv2d(32, 128, 5, stride=2, padding=2),
                                  nn.BatchNorm2d(128), nl,
                                  nn.Conv2d(128, 256, 5, stride=2, padding=2),
                                  nn.BatchNorm2d(256), nl,
                                  nn.Conv2d(256, 256, 5, stride=2, padding=2), nl)
        self.fc = nn.Sequential(nn.Linear(256 * 4 * 4, 1), nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('sigmoid'))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        return x, self.fc(x)
