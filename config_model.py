import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class LazyLinearWithConstraint(nn.LazyLinear):
    def __init__(self, *args, max_norm=1., **kwargs):
        super(LazyLinearWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return self(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def calculate_cosine_similarity(x, y):
    # 确保 x 和 y 的维度一致并且是三维 (batch_size, feature_dim_1, feature_dim_2)
    # 展开最后一个维度 128 以计算余弦相似度
    cos_sim = F.cosine_similarity(x, y, dim=-1)  # 在最后一个维度 128 上计算相似度

    # 返回二维结果 (batch_size, feature_dim_1)
    return cos_sim.detach().cpu().numpy()


# 绘制热力图函数
def plot_heatmap(data, title, xlabel, ylabel,i):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap='coolwarm', cbar=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.savefig('/root/autodl-tmp/model_picture/'+str(i)+'_CA_heatmap.png')




def plot_heat(temporal_features, frequency_features, i):
    # 取出 batch 中的一个样本进行计算, shape: [250, 128]
    temporal_sample = temporal_features[0]  # 取出第一个样本，shape为 [250, 128]
    frequency_sample = frequency_features[0]  # 同样取出一个样本，shape为 [250, 128]

    # 将特征点的特征向量提取出来，shape: [128, 250]，因为我们要计算特征维度之间的相似性
    temporal_features_128 = temporal_sample.T  # 转置，得到 [128, 250] 的张量，表示 128 维特征和 250 个特征点
    frequency_features_128 = frequency_sample.T  # 同样转置

    temporal_features_128 = temporal_features_128.detach().cpu().numpy()
    frequency_features_128 = frequency_features_128.detach().cpu().numpy()

    # 计算时空特征和频空特征在 128 维度上的余弦相似度，得到 [128, 128] 的相似矩阵
    similarity_matrix = cosine_similarity(temporal_features_128, frequency_features_128)

    # 绘制余弦相似性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm', cbar=True)
    if i == 1:
        plt.title("Cross-Covariance Attention - Similarity Before Feature Interaction")
    else:
        plt.title("Cross-Covariance Attention - Similarity After Feature Interaction")
    plt.xlabel("Frequency-Spatial Feature Dimensions")
    plt.ylabel("Temporal-Spatial Feature Dimensions")
    plt.savefig('/root/autodl-tmp/model_picture/' + str(i) + '_CCA_heatmap_new.png')
    plt.show()