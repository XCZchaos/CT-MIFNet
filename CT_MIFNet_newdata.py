import os

import mne
import numpy as np
import math
import random
import time
import scipy.io
import warnings
from sklearn.metrics import cohen_kappa_score
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from metric_newdata import GELU, Corr, LabelSmoothingLoss
from config_newdata import config

import torch
import torch.nn.functional as F
from config_model import Conv2dWithConstraint, LinearWithConstraint

from torch import nn
from sklearn.model_selection import train_test_split

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from commom_spatial_pattern_newdata import csp


gpus = [0, 1]    # gpus list
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))




class Inception(nn.Module):
    def __init__(self, emb_size: int = config.d_model, img_size1: int = config.C,
                 img_size2: int = config.T, patch_sizew: int = config.patchsize, patch_sizeh: int = config.patchsize,):
        super(Inception, self).__init__()
        self.patch_sizeh = patch_sizeh
        self.patch_sizew = patch_sizew


        self.unit1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 187), padding='same'),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )
        self.unit2 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 125), padding='same'),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )
        self.unit3 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 62), padding='same'),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )
        self.unit4 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 30), padding='same'),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )


        self.b_units = nn.ModuleList([self.unit1, self.unit2, self.unit3, self.unit4])
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            Conv2dWithConstraint(32, emb_size // 2, (8, 1), padding='same', max_norm=2.0),
            nn.BatchNorm2d(emb_size // 2),
            nn.LeakyReLU(0.5, inplace=True),
            Conv2dWithConstraint(emb_size // 2, emb_size, kernel_size=(self.patch_sizeh, self.patch_sizew),
                                 stride=(self.patch_sizeh, self.patch_sizew), padding=(0, 0), max_norm=2.0),
            nn.BatchNorm2d(emb_size),
            nn.LeakyReLU(0.3),
            nn.Conv2d(emb_size, emb_size, kernel_size=1, stride=1, padding=0),

            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.fc = LinearWithConstraint(8, 256, max_norm=2)
        self.fc2 = LinearWithConstraint(256, 64, max_norm=2)
        self.fc3 = LinearWithConstraint(64, 4, max_norm=2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 8))




    def forward(self, x):
        b1_outputs = [unit(x) for unit in self.b_units]

        b1_out = torch.cat(b1_outputs, dim=1)

        x = self.projection(b1_out)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.relu(x)


        return x



class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, max_len, d_model=config.d_model, dropout=config.p):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, device, in_channels: int = 32, patch_sizeh: int = config.patchsize,
                 patch_sizew: int = config.patchsize, emb_size: int = config.d_model, img_size1: int = config.C,
                 img_size2: int = config.T):
        super(PatchEmbedding, self).__init__()
        self.patch_sizeh = patch_sizeh
        self.patch_sizew = patch_sizew


        self.unit1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 187), padding='same'),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )
        self.unit2 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 125), padding='same'),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )
        self.unit3 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 62), padding='same'),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )
        self.unit4 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 30), padding='same'),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )


        self.b_units = nn.ModuleList([self.unit1, self.unit2, self.unit3, self.unit4])
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            Conv2dWithConstraint(32, emb_size // 2, (8, 1), padding='same', max_norm=2.0),
            nn.BatchNorm2d(emb_size // 2),
            nn.LeakyReLU(0.5, inplace=True),
            Conv2dWithConstraint(emb_size//2, emb_size, kernel_size=(self.patch_sizeh, self.patch_sizew),
                      stride=(self.patch_sizeh, self.patch_sizew), padding=(0, 0), max_norm=2.0),
            nn.BatchNorm2d(emb_size),
            nn.LeakyReLU(0.3),
            nn.Conv2d(emb_size, emb_size, kernel_size=1, stride=1, padding=0),

            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        # self.projection = nn.Sequential(
        #     # using a conv layer instead of a linear one -> performance gains
        #     Conv2dWithConstraint(1, emb_size, kernel_size=(self.patch_sizeh, self.patch_sizew),
        #                          stride=(self.patch_sizeh, self.patch_sizew), padding=(0, 0), max_norm=2.0),
        #     nn.BatchNorm2d(emb_size),
        #     nn.LeakyReLU(0.3),
        #     nn.Conv2d(emb_size, emb_size, kernel_size=1, stride=1, padding=0),
        #
        #     Rearrange('b e (h) (w) -> b (h w) e'),
        # )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

        self.positions = nn.Parameter(
            torch.randn(((img_size1 * img_size2) // (self.patch_sizew * self.patch_sizeh)), emb_size))
        self.nonpara = PositionalEncoding(((img_size1 * img_size2) // (self.patch_sizew * self.patch_sizeh))).to(device)

    def forward(self, x):
        b, _, _, _ = x.shape
        b1_outputs = [unit(x) for unit in self.b_units]

        b1_out = torch.cat(b1_outputs, dim=1)

        x = self.projection(b1_out)

        # x = self.projection(x)
        # cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # x += self.positions
        x = self.nonpara(x)
        return x



class CrossCovarianceAttention(nn.Module):
    def __init__(self, device, d_model, dim_k, dim_v, n_heads):
        super(CrossCovarianceAttention, self).__init__()
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.device = device

        self.q = nn.Linear(d_model, dim_k * n_heads)
        self.k = nn.Linear(d_model, dim_k * n_heads)
        self.v = nn.Linear(d_model, dim_v * n_heads)
        self.o = nn.Linear(dim_v * n_heads, d_model)
        self.norm_fact = 1 / (dim_k ** 0.5)
        self.dropout = nn.Dropout(config.p)



    def forward(self, x, y, requires_mask=True):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # size of x : [batch_size * seq_len * batch_size]

        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(y).reshape(-1, y.shape[0], y.shape[1], self.dim_k // self.n_heads)
        V = self.v(y).reshape(-1, y.shape[0], y.shape[1], self.dim_v // self.n_heads)




        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.norm_fact

        if requires_mask:
            mask = self.generate_mask(attention_scores.size()[3], attention_scores)
            attention_scores.masked_fill(mask, value=float("-inf"))

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        output = torch.matmul(attention_probs, V).transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        output = self.o(output)
        return output

    def generate_mask(self, dim, score):
        thre = torch.median(score, dim=-1, keepdim=True).values.to(self.device)
        thre = thre.expand_as(score)
        cha = score - thre
        one_vec = torch.ones_like(cha).to(self.device)
        zero_vec = torch.zeros_like(cha).to(self.device)

        mask = torch.where(cha > 0, zero_vec, one_vec).to(self.device)
        return mask == 1







class Mutihead_Attention(nn.Module):
    def __init__(self, device, d_model, dim_k, dim_v, n_heads):
        super(Mutihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads
        self.device = device

        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)
        # self.v = self.k

        self.o = nn.Linear(dim_v, d_model)
        self.norm_fact = 1 / math.sqrt(d_model)
        self.dropout = nn.Dropout(config.p)

    def generate_mask(self, dim, score):
        thre = torch.median(score, dim=-1, keepdim=True).values.to(self.device)
        thre = thre.expand_as(score)
        cha = score - thre
        one_vec = torch.ones_like(cha).to(self.device)
        zero_vec = torch.zeros_like(cha).to(self.device)

        mask = torch.where(cha > 0, zero_vec, one_vec).to(self.device)
        return mask == 1

    def forward(self, x, y, requires_mask=True):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # size of x : [batch_size * seq_len * batch_size]

        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1],
                              self.dim_k // self.n_heads)  # n_heads * batch_size * seq_len * dim_k
        K = self.k(y).reshape(-1, y.shape[0], y.shape[1],
                              self.dim_k // self.n_heads)  # n_heads * batch_size * seq_len * dim_k
        V = self.v(y).reshape(-1, y.shape[0], y.shape[1],
                              self.dim_v // self.n_heads)  # n_heads * batch_size * seq_len * dim_v
        # print("Attention V shape : {}".format(V.shape))
        attention_score = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact
        if requires_mask:
            mask = self.generate_mask(attention_score.size()[3], attention_score)

            attention_score.masked_fill(mask, value=float("-inf"))
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)  # 应用Dropout
        output = torch.matmul(attention_score, V).reshape(x.shape[0], x.shape[1], -1)
        # print("Attention output shape : {}".format(output.shape))

        output = self.o(output)
        return output






class Feed_Forward1(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, dropout_rate=0.3):
        super(Feed_Forward1, self).__init__()
        self.L1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.gelu = GELU().to(device)
        self.dropout = nn.Dropout(dropout_rate).to(device)
        self.L2 = nn.Linear(hidden_dim, input_dim).to(device)

    def forward(self, x):
        x = self.L1(x)
        x = self.gelu(x)
        x = self.dropout(x)  # 应用Dropout
        return self.L2(x)



class Feed_Forward(nn.Module):  # output
    def __init__(self, device, input_dim=config.d_model, hidden_dim=config.hidden):
        super(Feed_Forward, self).__init__()
        F1 = 16
        self.conv1 = nn.Conv2d(1, F1, (15, 16), bias=False, stride=(15, 16))  # Conv2d #F1*4*8
        self.dropout = nn.Dropout(config.p)
        self.gelu = GELU().to(device)

    def forward(self, x):
        output = self.gelu(self.conv1(x.unsqueeze(1)))
        output = self.dropout(output)
        output = output.contiguous().view(-1, self.num_flat_features(output))
        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features = num_features * s
        return num_features


class Add_Norm(nn.Module):
    def __init__(self, device, dropout_rate=0.3):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(dropout_rate).to(device)
        self.norm = nn.LayerNorm(128).to(device)  # 注意更新这里的LayerNorm以适应具体尺寸

    def forward(self, x, sub_layer, **kwargs):
        sub_output = sub_layer(x, **kwargs)
        x = x + sub_output
        x = self.dropout(x)  # 应用Dropout
        return self.norm(x)


class Encoder(nn.Module):
    def __init__(self, device, dim_seq, dim_fea, n_heads, hidden):
        super(Encoder, self).__init__()
        self.dim_seq = dim_seq
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden

        self.muti_atten = CrossCovarianceAttention(device, self.dim_fea, self.dim_k, self.dim_v, self.n_heads).to(device)
        self.feed_forward = Feed_Forward1(device, self.dim_fea, self.hidden).to(device)
        self.add_norm = Add_Norm(device).to(device)

    def forward(self, x):
        output = self.add_norm(x, self.muti_atten, y=x)
        output = self.add_norm(output, self.feed_forward)
        return output


class Encoder_last(nn.Module):
    def __init__(self, device, dim_seq, dim_fea, n_heads, hidden):
        super(Encoder_last, self).__init__()
        self.dim_seq = dim_seq
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden

        self.muti_atten = CrossCovarianceAttention(device, self.dim_fea, self.dim_k, self.dim_v, self.n_heads).to(device)
        self.feed_forward = Feed_Forward(device, self.dim_fea, self.hidden).to(device)
        self.add_norm = Add_Norm(device).to(device)

    def forward(self, x):
        output = self.add_norm(x, self.muti_atten, y=x)
        output = self.feed_forward(output)
        return output


class Decoder(nn.Module):
    def __init__(self, device, dim_seq, dim_fea, n_heads, hidden):
        super(Decoder, self).__init__()
        self.dim_seq = dim_seq
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden

        self.muti_atten = CrossCovarianceAttention(device, self.dim_fea, self.dim_k, self.dim_v, self.n_heads).to(device)
        self.feed_forward = Feed_Forward1(device, self.dim_fea, self.hidden).to(device)
        self.add_norm = Add_Norm(device).to(device)

    def forward(self, q, v):
        output = self.add_norm(v, self.muti_atten, y=q, requires_mask=True)
        output = self.add_norm(output, self.feed_forward)
        output = output + v
        return output


class Cross_modal(nn.Module):
    def __init__(self, device):
        super(Cross_modal, self).__init__()
        self.cross1 = Decoder(device, config.H * config.W, config.d_model, config.n_heads, config.hidden).to(device)
        self.cross2 = Decoder(device, config.H * config.W, config.d_model, config.n_heads, config.hidden).to(device)
        self.fc1 = nn.Linear(2 * config.d_model, config.d_model).to(device)

    def forward(self, target, f1):
        re = self.cross1(f1, target)
        return re


class Cross_modalto(nn.Module):
    def __init__(self, device, dim_seq=4 * config.H * config.W, dim_fea=config.d_model, n_heads=4,
                 hidden=config.hidden):
        super(Cross_modalto, self).__init__()
        self.dim_seq = dim_seq
        self.long = config.H * config.W
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden

        self.muti_atten = CrossCovarianceAttention(device, self.dim_fea, self.dim_k, self.dim_v, self.n_heads).to(device)
        self.feed_forward = Feed_Forward(device, self.dim_fea, self.hidden).to(device)
        self.add_norm = Add_Norm(device=device).to(device)

    def forward(self, q, v):
        output = self.add_norm(v, self.muti_atten, y=q, requires_mask=True)
        output = output + v
        output = self.feed_forward(output)
        return output




class Transformer_layer(nn.Module):
    def __init__(self, device, dmodel=config.d_model, num_heads=config.n_heads, num_tokens=config.H * config.W):
        super(Transformer_layer, self).__init__()
        self.encoder = Encoder(device, num_tokens, dmodel, num_heads, config.hidden).to(device)


    def forward(self, x):
        encoder_output = self.encoder(x) + x
        return encoder_output


class Transformer_layer_last(nn.Module):
    def __init__(self, device):
        super(Transformer_layer_last, self).__init__()
        self.encoder = Encoder_last(device, config.H * config.W, config.d_model, config.n_heads, config.hidden).to(
            device)

    def forward(self, x):
        encoder_output = self.encoder(x)
        return encoder_output




class Transformer(nn.Module):
    def __init__(self, device='cuda', depth=3, classes=2):
        super(Transformer, self).__init__()

        self.classes = classes
        # Patch Embedding
        self.embedding_t = PatchEmbedding(device=device, patch_sizeh=config.patchsizeth,
                                          patch_sizew=config.patchsizetw).to(device)
        self.embedding_f = PatchEmbedding(device=device, patch_sizeh=config.patchsizefh,
                                          patch_sizew=config.patchsizefw).to(device)
        self.norm = nn.LayerNorm(config.T).to(device)

        # Encoder
        self.model = nn.Sequential(*[Transformer_layer(device) for _ in range(depth)]).to(device)

        self.model_last = Cross_modalto(device).to(device)

        # cross-modal
        self.t = Cross_modal(device).to(device)
        self.f = Cross_modal(device).to(device)
        self.t1 = Cross_modal(device).to(device)
        self.f1 = Cross_modal(device).to(device)
        self.fc1 = nn.Linear(config.d_model * 2, config.d_model)
        self.fc2 = nn.Linear(1000, 500)
        self.fc = nn.Sequential(
            # LinearWithConstraint(4224, 3200, max_norm=2.0),
            # nn.ELU(),
            # nn.Dropout(0.5),
            LinearWithConstraint(4224, 256, max_norm=2.0),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, classes)
        )


    # CCA Block Method
    def forward(self, raw, fre):
        # print(raw.shape)
        # print('======================')
        # print(fre.shape)
        x_t1 = self.embedding_t(raw)
        x_f1 = self.embedding_f(self.norm(fre))



        # temporal data encoder
        x_t = self.model(x_t1)
        # frequency data encoder
        x_f = self.model(x_f1)
        # cross-modal

        x_t2 = self.t(x_t, x_f)
        x_f2 = self.f(x_f, x_t)

        x_t1 = self.t(x_t2, x_f2)
        x_f1 = self.f(x_f2, x_t2)

        x_t2 = x_t + x_t1
        x_f2 = x_f + x_f1

        x = torch.cat((x_f2, x_t2), axis=1)

        output = self.model_last(x, self.fc1(torch.cat((x_f2, x_t2), axis=-1)))
        output = self.fc(output)
        # x = torch.cat((x_f2, x_t2), axis=1)

        # x = self.fc2(x.transpose(-1, -2)).transpose(-1, -2)

        # output = self.model_last(x, self.fc1(torch.cat((x_f2, x_t2), axis=-1)))
        # output = self.fc(output)
        return output, x_t2, x_f2




class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res

        return x


class TFCformer(nn.Module):
    def __init__(self):
        super().__init__()



        self.block2 = Transformer()
        # self.block2 = Inception()

    def forward(self, x):
        x1 = x
        x_f = Corr(x1)
        output, x_t2, x_f2 = self.block2(x1, x_f)
        # output = self.block2(x)
        return output, x_t2, x_f2



class Trans():
    def __init__(self, nsub):
        super(Trans, self).__init__()
        self.batch_size = 50
        self.n_epochs = 200
        self.img_height = 22
        self.img_width = 600
        self.channels = 1
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.9
        self.nSub = nsub
        self.start_epoch = 0
        self.root = self.root = '/root/autodl-tmp/data/CAS_dataset/'  # the path of data

        self.pretrain = False



        self.img_shape = (self.channels, self.img_height, self.img_width)  # something no use

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion = LabelSmoothingLoss(2, config.smooth).to('cuda')

        self.model = TFCformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()


        # summary(self.model, (1, 16, 1000))

        self.centers = {}

    def get_event_data(self):
        """
        The filename is your subject file not a directory
        The data type is numpy.ndarray
        """
        if self.nSub < 10:
            filename = self.root + 'sub-00%d_29ByANT.set' % self.nSub
        else:
            filename = self.root + 'sub-0%d_29ByANT.set' % self.nSub
        epoch = mne.io.read_epochs_eeglab(filename)
        # epoch.resample(250)
        # standardize
        self.data = epoch.get_data(copy=True)
        # data = np.expand_dims(data, axis=1)
        target_mean = np.mean(self.data)
        target_std = np.std(self.data)
        self.data = (self.data - target_mean) / target_std
        self.label = epoch.events[:, -1]
        newdata = np.transpose(self.data, (0, 2, 1))
        Wb = csp(newdata, self.label - 1)  # common spatial pattern
        self.data = np.expand_dims(self.data, axis=1)
        self.data = np.einsum('abcd, ce -> abed', self.data, Wb)
        self.data = self.data[:, :, :, 1000:]

        # self.data = self.data[:, :, :, 1000:]



        return self.data, self.label




    def get_source_data(self):

        # to get the data of target subject
        self.total_data = scipy.io.loadmat(self.root + 'A0%dprocessing_T.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']

        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]

        # test data
        # to get the data of target subject

        self.test_tmp = scipy.io.loadmat(self.root + 'sub-00%d_29ByANT' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        # self.train_data = self.train_data[250:1000, :, :]
        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std


        tmp_alldata = np.transpose(np.squeeze(self.allData), (0, 2, 1))
        tmp_alllabel = np.transpose(np.squeeze(self.train_label))

        Wb = csp(tmp_alldata, self.allLabel - 1)  # common spatial pattern
        self.allData = np.einsum('abcd, ce -> abed', self.allData, Wb)
        self.testData = np.einsum('abcd, ce -> abed', self.testData, Wb)


        return self.allData, self.allLabel, self.testData, self.testLabel

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Do some data augmentation is a potential way to improve the generalization ability
    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(2):
            cls_idx = np.where(label == cls4aug)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 8, 2000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(4):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 4)
                    tmp_aug_data[ri, :, :, rj * 500:(rj + 1) * 500] = tmp_data[rand_idx[rj], :, :,
                                                                          rj * 500:(rj + 1) * 500]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        return aug_data, aug_label


    def calculate_kappa(self, y_true, y_pred):
        kappa = cohen_kappa_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        return kappa

    def train(self):
        data, label = self.get_event_data()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        bestAcc_all_folds = []
        averAcc_all_folds = []
        kappa_all_folds = []
        Y_true_all_folds = []
        Y_pred_all_folds = []
        fold_times = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.data, self.label)):
            print(f"Training fold {fold + 1}...")

            # 记录每个fold的开始时间
            start_time = time.time()

            # 重新初始化模型
            self.model = TFCformer().cuda()
            self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
            self.model = self.model.cuda()

            train_data, val_data = self.data[train_idx], self.data[val_idx]
            train_label, val_label = self.label[train_idx], self.label[val_idx]

            train_data = torch.from_numpy(train_data).float().cuda()
            train_label = torch.from_numpy(train_label - 1).long().cuda()
            dataset = TensorDataset(train_data, train_label)
            dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

            val_data = torch.from_numpy(val_data).float().cuda()
            val_label = torch.from_numpy(val_label - 1).long().cuda()
            val_dataset = TensorDataset(val_data, val_label)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

            bestAcc = 0
            averAcc = 0
            num = 0
            Y_true = torch.tensor([]).cuda()
            Y_pred = torch.tensor([]).cuda()

            total_step = len(dataloader)
            curr_lr = self.lr

            for e in range(self.n_epochs):
                self.model.train()
                if e % 40 == 0 and e > 0:
                    self.lr = self.lr * 0.8
                for i, (img, label) in enumerate(dataloader):
                    img = Variable(img.cuda())
                    label = Variable(label.cuda())
                    aug_data, aug_label = self.interaug(train_data.cpu().numpy(), train_label.cpu().numpy())

                    aug_data = torch.from_numpy(aug_data).float().cuda()
                    aug_label = torch.from_numpy(aug_label).long().cuda()

                    # img = torch.cat((img, aug_data))
                    # label = torch.cat((label, aug_label))
                    outputs, x_t2, x_f2 = self.model(img)
                    # outputs = self.model(img)

                    loss = self.criterion(outputs, label)
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if (e + 1) % 1 == 0:
                    self.model.eval()
                    outputs_test, x_t2, x_f2 = self.model(val_data)
                    # outputs_test = self.model(val_data)
                    loss_test = self.criterion(outputs_test, val_label)
                    y_pred = torch.max(outputs_test, 1)[1]
                    acc = float((y_pred == val_label).cpu().numpy().astype(int).sum()) / float(val_label.size(0))
                    train_pred = torch.max(outputs, 1)[1]
                    train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                    print(f'Fold {fold + 1}, Epoch: {e}, Train loss: {loss.detach().cpu().numpy()}, '
                          f'Test loss: {loss_test.detach().cpu().numpy()}, Train accuracy: {train_acc}, '
                          f'Test accuracy: {acc}')

                    num = num + 1
                    averAcc = averAcc + acc
                    if acc > bestAcc:
                        bestAcc = acc
                        Y_true = val_label
                        Y_pred = y_pred

            # 记录每个fold的结束时间
            end_time = time.time()
            fold_time = end_time - start_time
            fold_times.append(fold_time)

            averAcc = averAcc / num
            kappa = self.calculate_kappa(Y_true, Y_pred)

            bestAcc_all_folds.append(bestAcc)
            averAcc_all_folds.append(averAcc)
            kappa_all_folds.append(kappa)
            Y_true_all_folds.append(Y_true)
            Y_pred_all_folds.append(Y_pred)

            print(f'Fold {fold + 1} complete: Best Accuracy: {bestAcc}, Kappa: {kappa}')

        # 获取最短的时间
        min_fold_time = min(fold_times)

        print('Cross-validation complete.')
        print('Average best accuracy across folds:', np.max(bestAcc_all_folds))
        print('Average accuracy across folds:', np.mean(averAcc_all_folds))
        print('Average kappa across folds:', np.max(kappa_all_folds))

        return bestAcc_all_folds, averAcc_all_folds, Y_true_all_folds, Y_pred_all_folds, kappa_all_folds, min_fold_time

    def write_best_acc_to_file(self, sub, bestAcc, averAcc, kappa_all_folds, min_fold_time,
                               filename='/root/autodl-tmp/paper_model/TFCformer_best_acc.txt'):
        with open(filename, 'a') as f:
            f.write(f'Subject{sub}: Best Accuracy = {bestAcc:.6f}, Avg Accuracy = {averAcc:.6f}, '
                    f'kappa = {kappa_all_folds:.6f}, Shortest Time = {min_fold_time:.2f} seconds\n')


def main():
    best = 0
    aver = 0
    arr_list = [1, 3, 5, 8, 10, 12, 15, 18, 28]


    # for i in range(0, 29):
    for i in arr_list:
        seed_n = np.random.randint(500)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        # print('Subject %d' % (i+1))
        print('Subject %d' % (i))
        # trans = Trans(i + 1)
        trans = Trans(i)
        bestAcc, averAcc, Y_true, Y_pred, kappa, min_fold_time = trans.train()
        trans.write_best_acc_to_file(i+1, np.max(bestAcc), np.mean(averAcc), np.max(kappa), min_fold_time)
        print('THE BEST ACCURACY IS ' + str(np.max(bestAcc)))




if __name__ == "__main__":
    main()





