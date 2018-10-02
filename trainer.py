import torch
from PIL import Image
import torchvision
import torchvision.transforms as T
from model import *
from losses import *


class Trainer(object):
    def __init__(self, device, dataloader, enc, dec, dis, optimEnc,
                 optimDec, optimDis, loss_prior, loss_log_likelihood,
                 loss_gan_gen, loss_gan_dis, encoding_dim, gamma,
                 epoch, checkpoint, sample_size, sample_path, recon_img, recon_path):
        self.device = device
        self.dataloader = dataloader

        self.encoder = enc
        self.decoder = dec
        self.discriminator = dis

        self.optimEnc = optimEnc
        self.optimDec = optimDec
        self.optimDis = optimDis

        self.kl_loss = loss_prior
        self.loss_llike = loss_log_likelihood
        self.loss_gan_gen = loss_gan_gen
        self.loss_gan_dis = loss_gan_dis

        self.start_epoch = 0
        self.num_epochs = epoch
        self.encoding_dim = encoding_dim
        self.gamma = gamma

        self.pth = checkpoint
        
        self.sample_path = sample_path
        self.test_z = torch.randn(sample_size, encoding_dim, device=device)

        self.recon_path = recon_path
        img = Image.open(recon_img)
        transform = T.Compose([T.Scale((64, 64)), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.recon_img = transform(img).unsqueeze(0).to(device)

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

    def reconstruct(self, epoch):
        z, _, _ = self.encoder(self.recon_img)
        torchvision.utils.save_image(self.decoder(z).squeeze(0), self.recon_path + '/epoch%d.png' % epoch)

    def random_sample(self, epoch):
        torchvision.utils.save_image(self.decoder(self.test_z), self.sample_path + '/epoch%d.png' % epoch, nrow=8)

    def train_model(self):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        for epoch in range(self.start_epoch, self.num_epochs):
            print("Running Epoch {}".format(epoch + 1))
            l_enc = 0.0
            l_dec = 0.0
            l_dis = 0.0
            for i, data in enumerate(self.dataloader, 1):
                x,_ = data
                x = x.to(self.device)

                z, mu, logvar = self.encoder(x)
                x_recon = self.decoder(z)
                dl_original, d_original = self.discriminator(x)
                dl_recon, d_recon = self.discriminator(x_recon.detach())
                zp = torch.randn(x.size(0), self.encoding_dim, device=self.device)
                xp = self.decoder(zp)
                _, d_generated = self.discriminator(xp.detach())

                # TODO: Confirm that no unwanted gradient accumulation occurs
                L_prior = self.kl_loss(mu, logvar)
                L_llike = self.loss_llike(dl_original, dl_recon)
                L_generator = self.loss_gan_gen(d_recon, d_generated)
                L_discriminator = self.loss_gan_dis(d_original, d_recon, d_generated)

                L_encoder = L_prior + L_llike
                L_decoder = self.gamma * L_llike + L_generator

                self.optimEnc.zero_grad()
                #L_encoder = self.kl_loss(mu, logvar) + self.loss_llike(dl_original, dl_recon)
                L_encoder.backward(retain_graph=True)
                self.optimEnc.step()
                l_enc += L_encoder.item()

                self.optimDec.zero_grad()
                #L_decoder = self.gamma * self.loss_llike(dl_original, dl_recon) + self.loss_gan_gen(d_recon, d_generated)
                L_decoder.backward(retain_graph=True)
                self.optimDec.step()
                l_dec += L_decoder.item()

                self.optimDis.zero_grad()
                #L_discriminator = self.loss_gan_dis(d_original, d_recon, d_generated)
                L_discriminator.backward()
                self.optimDis.step()
                l_dis = L_discriminator.item()
            
            self.save_model(epoch)
            self.encoder.eval()
            self.decoder.eval()
            self.discriminator.eval()
            print('Epoch {} => Mean Encoder Loss : {} Mean Decoder Loss {} Mean Discriminator Loss {}'.format(epoch + 1, l_enc/i, l_dec/i, l_dis/i))
            self.reconstruct(epoch)
            self.random_sample(epoch)
            self.encoder.train()
            self.decoder.train()
            self.discriminator.train()
