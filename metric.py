from __future__ import division
from __future__ import print_function
import numpy as np
from config import config
import torch
from torch.autograd import Variable

import torch.nn as nn

import torch.nn.functional as F

from scipy.fftpack import fft, fftshift, ifft
from scipy.signal import stft
# import pywt


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        # return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))
        # return 0.5*x*(1.0 + torch.erf(x / torch.sqrt(2.0)))
        return F.relu(x)


def Corr(Raw):
    n_sam = Raw.size(0)
    Raw = Raw.cpu()
    Raw = Raw.data.numpy()
    Raw = np.squeeze(Raw)

    fft_matrix = np.abs(fft(Raw, n=config.fftn, axis=-1))
    FFT_matrix = np.concatenate((fft_matrix[:, :, :config.T // 2], fft_matrix[:, :, config.fftn - config.T // 2:]),
                                axis=-1)

    FFT_matrix = torch.FloatTensor(FFT_matrix)
    FFT_matrix = FFT_matrix
    FFT_matrix = FFT_matrix.unsqueeze(1).to('cuda')


    return FFT_matrix


def STFT_Corr(Raw):
    # Move input data from GPU to CPU and convert to NumPy array
    Raw = Raw.cpu().numpy()

    # Remove redundant dimensions (assuming Raw shape is (batch_size, 1, channel, timepoint))
    Raw = np.squeeze(Raw)  # Becomes (batch_size, channel, timepoint)

    # Compute STFT on the last dimension
    f, t, Zxx = stft(Raw, nperseg=config.nperseg, noverlap=config.noverlap, nfft=config.fftn, axis=-1)
    stft_matrix = np.abs(Zxx)

    # Get the time dimension of STFT results and ensure it's even
    if stft_matrix.shape[-1] % 2 != 0:
        stft_matrix = stft_matrix[..., :-1]  # If odd, truncate the last time step

    # Concatenation process (assuming concatenation is needed on the last dimension)
    mid_idx = stft_matrix.shape[-1] // 2
    STFT_matrix = np.concatenate((stft_matrix[..., :mid_idx], stft_matrix[..., mid_idx:]), axis=-1)

    # Convert to Torch tensor and move to CUDA
    STFT_matrix = torch.FloatTensor(STFT_matrix).to('cuda')
    print(STFT_matrix.shape)

    return STFT_matrix





def att_norm(att):
    mx = torch.ones((att.size(2), 1))
    att_sum = torch.matmul(torch.abs(att[0]), mx)
    att_sum1 = torch.matmul(att_sum, mx.T).unsqueeze(0)
    return att / att_sum1


class LabelSmoothingLoss(nn.Module):
    "Implement label smoothing."

    def __init__(self, class_num=2, smoothing=config.smooth):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.class_num = class_num

    def forward(self, x, target):

        assert x.size(1) == self.class_num
        if self.smoothing == 0:
            return nn.CrossEntropyLoss()(x, target)

        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.class_num - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        logprobs = F.log_softmax(x, dim=-1)
        mean_loss = torch.mean(torch.sum(-true_dist * logprobs, dim=-1))

        return mean_loss


class LabelSmoothingLoss1(nn.Module):

    def __init__(self, size=2, smoothing=config.smooth):
        super(LabelSmoothingLoss1, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))

        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        x = F.log_softmax(x, dim=-1)
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def KLloss(token1, token2):
    logp_x = F.log_softmax(token1, dim=-1)
    p_y = F.softmax(token2, dim=-1)
    kl_mean1 = F.kl_div(logp_x, p_y, reduction='batchmean')

    logp_y = F.log_softmax(token2, dim=-1)
    p_x = F.softmax(token1, dim=-1)
    kl_mean2 = F.kl_div(logp_y, p_x, reduction='batchmean')
    return kl_mean1



