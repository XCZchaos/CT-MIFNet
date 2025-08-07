import os


class Config(object):
    def __init__(self):
        # all
        self.N = 1
        self.p = 0.3
        self.d_model = 80  # 128
        self.hidden = self.d_model * 4
        self.n_heads = 4  # feature % n_heads**2 ==0

        # raw
        self.C = 8    # 64
        self.T = 1000    # 248
        self.patchsize = 8

        self.H = self.C // self.patchsize
        self.W = self.T // self.patchsize

        # frequence
        self.fs = 250
        self.Cf = 8      # 64
        self.Tf = 1000      # 248
        self.patchsizefh = 8  # 64
        self.patchsizefw = 4
        self.fftn = 1000

        self.Hf = self.Cf // self.patchsizefh
        self.Wf = self.Tf // self.patchsizefw

        # temporal
        self.Ct = 8      # 64
        self.Tt = 1000      # 1000
        self.patchsizeth = 8   # 64
        self.patchsizetw = 4
        self.Ht = self.Ct // self.patchsizeth
        self.Wt = self.Tt // self.patchsizetw

        self.batchsize = 64
        self.epoch = 110
        self.patience = 100
        self.lr = 5e-4
        self.val = 2240
        self.smooth = 0.01
        self.num_class = 2
        self.kl = 0.001

        self.sample = 1


config = Config()