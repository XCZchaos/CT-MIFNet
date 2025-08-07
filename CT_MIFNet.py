import os
import numpy as np
import math
import random
import time
import scipy.io
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import cohen_kappa_score
from torch.autograd import Variable
from metric import GELU, Corr, LabelSmoothingLoss, STFT_Corr
from config import config
from config_model import Conv2dWithConstraint, LinearWithConstraint, calculate_cosine_similarity, plot_heat

import torch
import torch.nn.functional as F

from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from commom_spatial_pattern import csp


# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

gpus = [0, 1]    # gpus list
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))



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
        # print(self.pe.shape)
        # print(Variable(self.pe[:, :x.size(1)], requires_grad=False).shape)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)



class PatchEmbedding(nn.Module):
    def __init__(self, device, patch_sizeh: int = config.patchsize,
                 patch_sizew: int = config.patchsize, emb_size: int = config.d_model, img_size1: int = config.C,
                 img_size2: int = config.T, fs:int = config.fs):
        super(PatchEmbedding, self).__init__()
        self.patch_sizeh = patch_sizeh
        self.patch_sizew = patch_sizew


        self.unit1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, int(0.64*fs)), padding='same'),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )
        self.unit2 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, int(0.32*fs)), padding='same'),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )
        self.unit3 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, int(0.16*fs)), padding='same'),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )
        self.unit4 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, int(0.08*fs)), padding='same'),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )

        self.max_unit = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 30), padding='same'),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3)
        )


        self.b_units = nn.ModuleList([self.unit1, self.unit2, self.unit3, self.unit4])
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            Conv2dWithConstraint(16, emb_size // 2, (16, 1), padding='same', max_norm=2.0),               #
            nn.BatchNorm2d(emb_size // 2),
            nn.LeakyReLU(0.5, inplace=True),
            Conv2dWithConstraint(emb_size//2, emb_size, kernel_size=(self.patch_sizeh, self.patch_sizew),
                      stride=(self.patch_sizeh, self.patch_sizew), padding=(0, 0), max_norm=2.0),
            nn.BatchNorm2d(emb_size),
            nn.LeakyReLU(0.3),
            nn.Conv2d(emb_size, emb_size, kernel_size=1, stride=1, padding=0),
            Rearrange('b e (h) (w) -> b (h w) e'),

        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

        self.positions = nn.Parameter(
            torch.randn(((img_size1 * img_size2) // (self.patch_sizew * self.patch_sizeh)), emb_size))
        self.nonpara = PositionalEncoding(((img_size1 * img_size2) // (self.patch_sizew * self.patch_sizeh))).to(device)

    def forward(self, x):
        b, _, _, _ = x.shape
        b1_outputs = [unit(x) for unit in self.b_units]

        b1_out = torch.cat(b1_outputs, dim=1)
        # b1_out = self.max_unit(x)

        x = self.projection(b1_out)
        # x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # x += self.positions
        x = self.nonpara(x)
        return x




class Mutihead_Attention(nn.Module):
    """
    The implementation of the multi-head attention mechanism is the same as the traditional attention mechanism,
    except that the median mask is used for masking.
    """
    def __init__(self, device, d_model, dim_k, dim_v, n_heads):
        super(Mutihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads
        self.device = device

        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)
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

        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(y).reshape(-1, y.shape[0], y.shape[1], self.dim_k // self.n_heads)
        V = self.v(y).reshape(-1, y.shape[0], y.shape[1], self.dim_v // self.n_heads)

        attention_score = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact
        if requires_mask:
            mask = self.generate_mask(attention_score.size()[3], attention_score)
            attention_score.masked_fill(mask, value=float("-inf"))
        attention_score = F.softmax(attention_score, dim=-1)

        # -----------------------------------------------------------
        attention_score = self.dropout(attention_score)
        output = torch.matmul(attention_score, V).reshape(x.shape[0], x.shape[1], -1)

        output = self.o(output)
        return output



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
        self.temperature = nn.Parameter(torch.ones(n_heads*2, 1, 1))

    def forward(self, x, y, requires_mask=True):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # x, y shape: [batch_size, seq_len, d_model]

        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(y).reshape(-1, y.shape[0], y.shape[1], self.dim_k // self.n_heads)
        V = self.v(y).reshape(-1, y.shape[0], y.shape[1], self.dim_v // self.n_heads)



        Q_normalized = torch.nn.functional.normalize(Q, dim=-1)
        K_normalized = torch.nn.functional.normalize(K, dim=-1)



        cross_cov_matrix = torch.matmul(Q_normalized.transpose(-2, -1), K_normalized)
        attention_scores = cross_cov_matrix * self.norm_fact

        if requires_mask:
            mask = self.generate_mask(attention_scores.size(-1), attention_scores)
            attention_scores.masked_fill(mask, value=float("-inf"))

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)



        attention_output = torch.matmul(attention_probs, V.transpose(-2, -1)).transpose(-2, -1)

        attention_output = attention_output.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)


        output = self.o(attention_output)
        return output

    def generate_mask(self, dim, score):
        thre = torch.median(score, dim=-1, keepdim=True).values.to(self.device)
        thre = thre.expand_as(score)
        cha = score - thre
        one_vec = torch.ones_like(cha).to(self.device)
        zero_vec = torch.zeros_like(cha).to(self.device)

        mask = torch.where(cha > 0, zero_vec, one_vec).to(self.device)
        return mask == 1






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
        x = self.dropout(x)
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
        self.norm = nn.LayerNorm(128).to(device)

    def forward(self, x, sub_layer, **kwargs):

        sub_output = sub_layer(x, **kwargs)
        x = x + sub_output
        x = self.dropout(x)
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

        self.muti_atten =  Mutihead_Attention(device, self.dim_fea, self.dim_k, self.dim_v, self.n_heads).to(device)
        self.feed_forward = Feed_Forward1(device, self.dim_fea, self.hidden).to(device)
        self.add_norm = Add_Norm(device).to(device)

    def forward(self, x):
        output = self.add_norm(x, self.muti_atten, y=x)
        output = self.add_norm(output, self.feed_forward)
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







class Transformer(nn.Module):
    def __init__(self, device='cuda', depth=3, classes=4):
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

        self.fc1 = nn.Linear(config.d_model * 2, config.d_model)
        self.fc2 = nn.Linear(500, 250)
        self.fc = nn.Sequential(
            LinearWithConstraint(2048, 256, max_norm=2.0),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )


    def forward(self, raw, fre):
        x_t1 = self.embedding_t(raw)
        x_f1 = self.embedding_f(self.norm(fre))
        out_t1 = x_t1
        out_f1 = x_f1

        # temporal data encoder
        x_t = self.model(x_t1)
        # frequency data encoder
        x_f = self.model(x_f1)
        # cross-modal
        # 进行交叉计算 进行两次计算
        x_t2 = self.t(x_t, x_f)
        x_f2 = self.f(x_f, x_t)

        x_t1 = self.t(x_t2, x_f2)
        x_f1 = self.f(x_f2, x_t2)

        x_t2 = x_t + x_t1
        x_f2 = x_f + x_f1
        """
        Concatenating from different dimensions to enhance the ability to extract richer features, 
        improve the model's robustness, capture more complex features, and make the model more stable.
        """
        x = torch.cat((x_f, x_t), axis=1)
        x = self.fc2(x.transpose(-1, -2)).transpose(-1, -2)
        output = self.model_last(x, self.fc1(torch.cat((x_f, x_t), axis=-1)))
        output = self.fc(output)

        return output, x_t2, x_f2, out_t1, out_f1, x_t, x_f





class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res

        return x


class CT_MIFNet(nn.Module):
    def __init__(self):
        super().__init__()


        self.block2 = Transformer()


    def forward(self, x):

        x1 = x
        x_f = Corr(x1)
        output, x_t2, x_f2, out_t1, out_f1, x_t, x_f = self.block2(x1, x_f)


        return output, x_t2, x_f2, out_t1, out_f1, x_t, x_f



class Trans():
    def __init__(self, nsub):
        super(Trans, self).__init__()
        self.batch_size = 50
        self.n_epochs = 200
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.9
        self.nSub = 3
        self.start_epoch = 0
        self.root = '/root/autodl-tmp/data/BCICIV_2a_gdf_01/'  # the path of data
        self.pretrain = False
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.model = CT_MIFNet().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        self.centers = {}
        self.criterion = LabelSmoothingLoss(4, config.smooth).to('cuda')

    def get_source_data(self):
        self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']
        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)
        self.allData = self.train_data
        self.allLabel = self.train_label[0]
        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']
        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)
        self.testData = self.test_data
        self.testLabel = self.test_label[0]
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std
        tmp_alldata = np.transpose(np.squeeze(self.allData), (0, 2, 1))

        tmp_alllabel = np.transpose(np.squeeze(self.train_label))
        Wb = csp(tmp_alldata, self.allLabel - 1)
        self.allData = np.einsum('abcd, ce -> abed', self.allData, Wb)
        self.testData = np.einsum('abcd, ce -> abed', self.testData, Wb)
        return self.allData, self.allLabel, self.testData, self.testLabel

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def calculate_kappa(self, y_true, y_pred):
        kappa = cohen_kappa_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        return kappa

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 16, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :, rj * 125:(rj + 1) * 125]
            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]
        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label - 1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def train(self):
        img, label, test_data, test_label = self.get_source_data()
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)
        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))
        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        train_losses = []
        train_accuracies = []

        for e in range(self.n_epochs):
            in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))
                outputs, x_t2, x_f2, out_t1, out_f1, x_t, x_f = self.model(img)
                loss = self.criterion(outputs, label)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



            if (e + 1) % 1 == 0:
                self.model.eval()
                outputs_test, x_t2, x_f2, out_t1, out_f1, x_t, x_f = self.model(test_data)
                loss_test = self.criterion(outputs_test, test_label)
                y_pred = torch.max(outputs_test, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                print('Epoch:', e,
                      '  Train loss:', loss.detach().cpu().numpy(),
                      '  Test loss:', loss_test.detach().cpu().numpy(),
                      '  Train accuracy:', train_acc,
                      '  Test accuracy is:', acc)
                train_losses.append(loss.detach().cpu().numpy())
                train_accuracies.append(train_acc)
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

                    final_out_t2 = x_t2

                    final_out_f2 = x_f2


                if e == self.n_epochs - 1:
                    data_1 = calculate_cosine_similarity(x_f, x_t)
                    data_2 = calculate_cosine_similarity(x_f2, x_t2)
                    plot_heat(x_t, x_f, i=1)
                    plot_heat(x_t2, x_f2, i=2)

        torch.save(self.model.module.state_dict(), '/root/autodl-tmp/model_picture/TFCformer_model_Subject_%d.pth' % (self.nSub))
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        kappa = self.calculate_kappa(Y_true, Y_pred)
        print('The kappa score is:', kappa)
        return bestAcc, averAcc, Y_true, Y_pred, kappa



def main():
    best = 0
    aver = 0
    for i in range(9):
        seed_n = np.random.randint(500)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        print('Subject %d' % (i+1))
        trans = Trans(i + 1)
        bestAcc, averAcc, Y_true, Y_pred, kappa = trans.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))

        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


if __name__ == "__main__":
    main()

