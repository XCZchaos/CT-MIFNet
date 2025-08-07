import itertools

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.metrics import confusion_matrix
import torch

def csp(data_train, label_train):
    idx_0 = np.squeeze(np.where(label_train == 0))
    idx_1 = np.squeeze(np.where(label_train == 1))
    idx_2 = np.squeeze(np.where(label_train == 2))
    idx_3 = np.squeeze(np.where(label_train == 3))

    W = []
    for n_class in range(4):
        if n_class == 0:
            idx_L = idx_0
            idx_R = np.concatenate((idx_1, idx_2, idx_3))     # , idx_2, idx_3
        elif n_class == 1:
            idx_L = idx_1
            idx_R = np.concatenate((idx_0, idx_2, idx_3))     # , idx_2, idx_3
        elif n_class == 2:
            idx_L = idx_2
            idx_R = np.concatenate((idx_0, idx_1, idx_3))
        elif n_class == 3:
            idx_L = idx_3
            idx_R = np.concatenate((idx_0, idx_1, idx_2))

        idx_R = np.sort(idx_R)
        Cov_L = np.zeros([3, 3, len(idx_L)])
        Cov_R = np.zeros([3, 3, len(idx_R)])

        for nL in range(len(idx_L)):
            E = data_train[idx_L[nL], :, :]
            EE = np.dot(E.transpose(), E)
            Cov_L[:, :, nL] = EE / np.trace(EE)

        for nR in range(len(idx_R)):
            E = data_train[idx_R[nR], :, :]
            EE = np.dot(E.transpose(), E)
            Cov_R[:, :, nR] = EE / np.trace(EE)

        Cov_L = np.mean(Cov_L, axis=2)
        Cov_R = np.mean(Cov_R, axis=2)
        CovTotal = Cov_L + Cov_R

        # Check and replace infs and NaNs in covariance matrices
        Cov_L = np.nan_to_num(Cov_L, nan=0.0, posinf=0.0, neginf=0.0)
        Cov_R = np.nan_to_num(Cov_R, nan=0.0, posinf=0.0, neginf=0.0)
        CovTotal = np.nan_to_num(CovTotal, nan=0.0, posinf=0.0, neginf=0.0)

        lam, Uc = eig(CovTotal)
        eigorder = np.argsort(lam)[::-1]
        lam = lam[eigorder]
        Ut = Uc[:, eigorder]

        Ptmp = np.sqrt(np.diag(np.power(lam, -1)))
        P = np.dot(Ptmp, Ut.transpose())

        # Check and replace infs and NaNs in transformation matrices
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        SL = np.dot(P, Cov_L)
        SLL = np.dot(SL, P.transpose())
        SR = np.dot(P, Cov_R)
        SRR = np.dot(SR, P.transpose())

        # Check and replace infs and NaNs in scaled covariance matrices
        SLL = np.nan_to_num(SLL, nan=0.0, posinf=0.0, neginf=0.0)
        SRR = np.nan_to_num(SRR, nan=0.0, posinf=0.0, neginf=0.0)

        lam_R, BR = eig(SRR)
        erorder = np.argsort(lam_R)
        B = BR[:, erorder]

        w = np.dot(P.transpose(), B)

        # Check and replace infs and NaNs in the resulting filter matrix
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        W.append(w)

    Wb = np.concatenate((W[0][:, 0:2], W[1][:, 0:2], W[2][:, 0:2], W[3][:, 0:2]), axis=1)             # , W[2][:, 0:2], W[3][:, 0:2]

    return Wb











