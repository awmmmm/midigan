from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import Datasetnpy
from pianoGAN import *
import torch
import torch.autograd as autograd
import numpy as np
from tqdm import tqdm
from util.metric import AverageMeter
import os
import matplotlib.pyplot as plt



###hyperparameter
batchsize = 64


lambda_gp = 10
lambda_g_1 = 0.3
lambda_g_2 = 0.1

epoch_stop = 500
n_epochs = 500
threshhold = 0.015

###initialize
generator = generator().cuda()
discriminator = discriminator().cuda()
decoder = decoder().cuda()
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
decoder.apply(weights_init_normal)
dataset = Datasetnpy()
optimizer_G = torch.optim.Adam(generator.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=0.0002,betas=(0.5,0.999))
# optimizer_Dec = torch.optim.SGD(decoder.parameters(),lr=0.002,momentum=0.9)
optimizer_Dec = torch.optim.Adam(decoder.parameters(),lr=0.0002,betas=(0.9,0.999))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_Dec)
DEC_Loss = nn.MSELoss()
train_losses = []
val_losses = []
train_dec_losses = []
val_dec_losses = []
finetune_dec_loss = []
Tensor = torch.cuda.FloatTensor
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1,1 )))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    # print(gradients.norm(2, dim=1).shape)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty




###load data
train_db,val_db = torch.utils.data.random_split(dataset,[120000,38373])
# dataloader = DataLoader(
#     dataset = dataset,shuffle=True,pin_memory=True,
#     batch_size= batchsize,drop_last=False
# )
traindataloader = DataLoader(dataset = train_db,shuffle=True,pin_memory=True,batch_size=batchsize,drop_last=False,num_workers=2)
valdataloader = DataLoader(dataset = val_db,shuffle=True,pin_memory=True,batch_size=batchsize,drop_last=False,num_workers=2)




for epoch in range(n_epochs):
    if epoch < epoch_stop:

        train_loss_meter = AverageMeter()
        val_loss_meter = AverageMeter()
        train_dec_loss_meter = AverageMeter()
        val_dec_loss_meter = AverageMeter()
        loop_train = tqdm(enumerate(traindataloader), total=len(traindataloader))
        # loop_val = tqdm(enumerate(valdataloader), total=len(valdataloader))
        # ----------
        #  train
        # ----------
        for i , real_data in loop_train:

            real_data =real_data.float().cuda()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            generator.train()
            discriminator.train()
            decoder.train()
            for p in discriminator.parameters():
                p.requires_grad = True

            for p in generator.parameters():
                p.requires_grad = False

            for p in decoder.parameters():
                p.requires_grad = False

            optimizer_D.zero_grad()

            # z = Variable(Tensor(-1 + 2 * np.random.random((real_data.shape[0], 100))))
            z = Variable(Tensor(np.random.uniform(-1,1,(real_data.shape[0], 100))))
            fake_data = generator(z)
            real_data = real_data.view(fake_data.shape)
            # print(real_data.shape)
            real_validity = discriminator(real_data)

            fake_validity = discriminator(fake_data)
            # with torch.backends.cudnn.flags(enabled=False):
            gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_data.data)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty


            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            train_loss_meter.update(d_loss.item(),real_data.size(0))

            # f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            if i % 5 == 0:
        # -----------------
        #  Train Generator
        # -----------------
                for p in discriminator.parameters():
                    p.requires_grad = False

                for p in generator.parameters():
                    p.requires_grad = True

                for p in decoder.parameters():
                    p.requires_grad = False

                fake_data = generator(z)
                fake_validity = discriminator(fake_data)
                recovery_data = decoder(fake_data)
                # fake_data_2 = generator(recovery_data)
                # g_loss3 = DEC_Loss(fake_data_2, fake_data)#make G can generate fake data 2 similar to fake data
                g_loss1 = DEC_Loss(recovery_data, z)
                g_loss2 = -torch.mean(fake_validity)
                # g_loss = lambda_g_1*g_loss1+(1-lambda_g_1-lambda_g_2)*g_loss2+lambda_g_2*g_loss3
                # g_loss = lambda_g_1 * g_loss1 + (1 - lambda_g_1) * g_loss2
                g_loss = g_loss2
                g_loss.backward()
                optimizer_G.step()

            # -----------------
            #  Train Decoder
            # -----------------
            optimizer_Dec.zero_grad()

            for p in discriminator.parameters():
                p.requires_grad = False

            for p in generator.parameters():
                p.requires_grad = False

            for p in decoder.parameters():
                p.requires_grad =True

            fake_data = generator(z)
            recovery_data = decoder(fake_data)
            # assert recovery_data.shape == z.shape
            dec_loss = DEC_Loss(recovery_data,z)
            dec_loss.backward()
            optimizer_Dec.step()
            train_dec_loss_meter.update(dec_loss.item(),real_data.size(0))
            loop_train.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            loop_train.set_postfix(g_loss=g_loss.item(), d_loss=d_loss.item(), avg=train_loss_meter.avg,dec_loss_avg = train_dec_loss_meter.avg)

        train_losses.append(train_loss_meter.avg)
        train_dec_losses.append(train_dec_loss_meter.avg)
            # if i % 10 == 0:
            #     print(f'loss {train_loss_meter.val:.4f} ({train_loss_meter.avg:.4f})\t')
            # ----------
            #  evaluate
            # ----------
        loop_val = tqdm(enumerate(valdataloader), total=len(valdataloader))
        for i, real_data in loop_val:
            generator.eval()
            discriminator.eval()
            decoder.eval()
            real_data = real_data.float().cuda()

            z = Variable(Tensor(-1 + 2 * np.random.random((real_data.shape[0], 100))))
            fake_data = generator(z)
            recovery_data = decoder(fake_data)
            dec_loss = DEC_Loss(recovery_data, z)
            real_data = real_data.view(fake_data.shape)
            # print(real_data.shape)
            real_validity = discriminator(real_data)
            fake_validity = discriminator(fake_data)
            # with torch.backends.cudnn.flags(enabled=False):
            gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_data.data)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            val_loss_meter.update(d_loss.item(), real_data.size(0))
            val_dec_loss_meter.update(dec_loss.item(),real_data.size(0))

            loop_val.set_description(f'evaluate :Epoch [{epoch+1}/{n_epochs}]')
            loop_val.set_postfix(d_loss=d_loss.item(),avg=val_loss_meter.avg,dec_loss_avg =val_dec_loss_meter.avg)

        val_losses.append(val_loss_meter.avg)
        val_dec_losses.append(val_dec_loss_meter.avg)
        os.makedirs('./checkpoint_withno_eloss/{}'.format(epoch+1),exist_ok=True)
        torch.save(generator.state_dict(),'./checkpoint_withno_eloss/{}/G_{:.3f}_{:.3f}_{}.pth'.format(epoch+1,train_loss_meter.avg,val_loss_meter.avg,epoch+1))
        torch.save(optimizer_G.state_dict(),'./checkpoint_withno_eloss/{}/opt_G.pth'.format(epoch+1))
        torch.save(decoder.state_dict(),'./checkpoint_withno_eloss/{}/DEC_{:.3f}_{:.3f}_{}.pth'.format(epoch+1,train_dec_loss_meter.avg,val_dec_loss_meter.avg,epoch+1))
        torch.save(optimizer_Dec.state_dict(),'./checkpoint_withno_eloss/{}/opt_DEC.pth'.format(epoch+1))



    if epoch+1 == epoch_stop:#plot\save_model
        plt.plot(train_losses, c='red')
        plt.plot(val_losses, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['train', 'val'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('./checkpoint_withno_eloss/loss_RNN_4001e.png', transparent=True,dpi = 300)
        plt.close()
        train_loss_npy = np.array(train_losses)
        val_loss_npy = np.array(val_losses)
        np.save('./checkpoint_withno_eloss/train_loss.npy',train_loss_npy)
        np.save('./checkpoint_withno_eloss/val_loss.npy', val_loss_npy)
        train_dec_loss_npy = np.array(train_dec_losses)
        val_dec_loss_npy = np.array(val_dec_losses)
        np.save('./checkpoint_withno_eloss/train_dec_loss.npy', train_dec_loss_npy)
        np.save('./checkpoint_withno_eloss/val_dec_loss.npy', val_dec_loss_npy)