import torch
from model import *
from losses import *


class Trainer(object):
    def __init__(self, device, dataloader, enc, dec, dis, optimEnc,
                 optimDec, optimDis, loss_prior, loss_log_likelihood,
                 loss_gan_gen, loss_gan_dis, epoch, checkpoint):
        self.device = device
        self.encoder = enc
        self.decoder = dec
        self.dataloader = dataloader
        self.discriminator = dis
        self.optimEnc = optimEnc
        self.optimDec = optimDec
        self.optimDis = optimDis
        self.kl_loss = loss_prior
        self.loss_llike = loss_log_likelihood
        self.loss_gan_gen = loss_gan_gen
        self.loss_gan_dis = loss_gan_dis
        self.start_epoch = 0
        self.num_epochs = 0
        self.pth = checkpoint

    def save_model(self, epoch):
        torch.save({
            'epoch': epoch + 1,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimEnc': self.optimEnc.state_dict(),
            'optimDec': self.optimDec.state_dict(),
            'optimDis': self.optimDis.state_dict()},
            self.pth)

    def load_model(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.optimEnc.load_state_dict(checkpoint['optimEnc'])
            self.optimDec.load_state_dict(checkpoint['optimDec'])
            self.optimDis.load_state_dict(checkpoint['optimDis'])
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}'. Start Fresh Training".
                  format(self.checkpoints))
            self.start_epoch = 0
