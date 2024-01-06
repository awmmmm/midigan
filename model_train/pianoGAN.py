import torch
import torch.nn as nn
from copy import deepcopy
#

class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64,128,3,stride=1,padding =1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128,64,3,padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(64,1,3,padding=1),
            nn.Tanh()

        )
        self.linear = nn.Linear(100,25*64)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self,x):#B,100
        x = self.linear(x)
        x =x.view(x.shape[0],64,25)
        x = self.model(x)
        return x

# class discriminator(nn.Module):
#     def __init__(self):
#         super(discriminator,self).__init__()
#         # self.conv = nn.Conv1d()
#         self.lstm = nn.LSTM(input_size=1,hidden_size=64,num_layers=3,dropout=0.4,batch_first=True,bidirectional=True)
#         self.lin_proj = nn.Linear(128,1)
#     def forward(self,x):#B,100
#         x = x.view(-1,100,1)
#         x ,_= self.lstm(x)#B,100,1->B,100,128
#         # print(x.shape)
#         x = self.lin_proj(x)#B,100,256 ->B,100,1
#         x =torch.mean(x,dim=1)
#         return x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        # self.conv = nn.Conv1d()
        def decoder_block(in_filters, out_filters, ):
            block = [nn.Conv1d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), ]
            return block

        self.model = nn.Sequential(
            *decoder_block(1, 16),
            *decoder_block(16, 32),
            *decoder_block(32, 64),
            *decoder_block(64, 128),
        )
        # ds_size = 100 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128*7,1))
    def forward(self,x):#B,100
        # x = x.view(-1,100)
        x = self.model(x)#B,1,100->B,128,7
        # print(x.shape)
        x = x.view(-1,128*7)
        x = self.adv_layer(x)#B,128*7 ->B,1
        # x =torch.mean(x,dim=1)
        return x

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        def decoder_block(in_filters, out_filters, bn=True):
            # block = [nn.Conv1d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),nn.BatchNorm1d(out_filters)]
            block = [nn.Conv1d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                     ]
            return block
        self.model = nn.Sequential(
            *decoder_block(1,16),
            *decoder_block(16,32),
            *decoder_block(32,64),
            *decoder_block(64,128),
        )
        # ds_size = 100 // 2 ** 4
        ds_size = 7
        self.recover_layer = nn.Sequential(
            nn.Linear(128*ds_size,448),
            # nn.Dropout(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(448,224),
            # nn.Dropout(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(224,100),
            nn.Tanh())

    def forward(self,x):
        x =self.model(x)
        x = x.view(-1, 128 * 7)
        x = self.recover_layer(x)

        return x


