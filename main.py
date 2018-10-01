import torch
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as datasets
import torch.utils as utils
import model
import losses
import trainer

transforms = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
celebA = datasets.ImageFolder('./dataset', transform=transforms)
dataloader = utils.data.DataLoader(celebA, batch_size=64, shuffle=True, num_workers=4)

device = torch.device('cuda:0')

encoder = model.Encoder()
decoder = model.Decoder()
discriminator = model.Discriminator()
encoder.to(device)
decoder.to(device)
discriminator.to(device)

optimEnc = optim.Adam(encoder.parameters(), 0.0001)
optimDec = optim.Adam(decoder.parameters(), 0.0001)
optimDis = optim.Adam(discriminator.parameters(), 0.0001)
tr = trainer.Trainer(device, dataloader, encoder, decoder, discriminator, 
                     optimEnc, optimDec, optimDis,
                     losses.kl_loss, losses.log_loss, losses.decoder_minimax_loss, losses.discriminator_minimax_loss,
                     2048, 0.01, 300, 'model.obj', 16, './sample', '202599.jpg', './recon')
tr.train_model()
