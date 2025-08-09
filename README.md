# **CT-MIFNet: Convolutional Transformer-based Multi-View Interaction and Fusion Network for EEG Decoding [(paper)](https://www.sciencedirect.com/science/article/abs/pii/S1746809425009322)**

This repository provides the implementation of **CT-MIFNet**, a deep learning model based on CNN and Transformer, which enhances model performance by interacting with and fusing information from different views.

---

## **Features**
- A novel parallel dual-branch convolutional Transformer-based multi-view
interaction and fusion network is proposed for EEG decoding.

- A cross-covariance attention mechanism is introduced to facilitate feature
interaction and fusion across diverse perspectives while reducing computational
load.
- Extensive comparisons with the state-of-the-art models on three datasets with two
distinct BCI paradigms demonstrate the generalization and robustness of the
proposed model.

![model](Figure_01.png)
![model](Figure_04.png)
---



## **Usage**

You can directly run the model files TFCformer.py and TFCformer_2b.py. We will update the descriptions for some files shortly after our paper is accepted.

---
## **Citation**
Hope this code can be useful. I would appreciate you citing us in your paper. 
```
@article{XIONG2026108421,
title = {CT-MIFNet: Convolutional transformer-based multi-view interaction and fusion network for EEG decoding},
journal = {Biomedical Signal Processing and Control},
volume = {112},
pages = {108421},
year = {2026},
issn = {1746-8094},
doi = {https://doi.org/10.1016/j.bspc.2025.108421},
url = {https://www.sciencedirect.com/science/article/pii/S1746809425009322},
author = {Yibo Xiong and Jinming Li and Yun Zhuang and Xiangyue Zhao and Yilu Xu and Lilin Jie},
keywords = {Brain-computer interface, Motor Imagery, Pain Perception, Transformer, Multi-view Information, Feature Interaction and Fusion},
abstract = {Convolutional neural networks (CNNs) are effective at extracting local features but are limited in capturing long-term dependencies due to their fixed kernel size. In contrast, Transformers are capable of capturing long-range dependencies through the self-attention mechanism.Although there are frameworks that extract both local and global features by combining CNN with Transformer in brain-computer interface (BCI) systems, multi-view features have not been effectively explored in Electroencephalography (EEG) decoding. Moreover, the increased computational complexity introduced by the attention mechanism in Transformers poses challenges, hindering their application to EEG signals with long sequence. Therefore, a novel Convolutional Transformer-based multi-view Interaction and Fusion Network (CT-MIFNet) is proposed. Initially, the preprocessed EEG signals are passed through a spatial transformation module, which reduces dimensionality while minimizing noise. After undergoing fast fourier transform (FFT) and branching into two separate paths, the signals are input into a Patch Embedding module with multi-scale convolution mapping to extract temporal, frequency, and spatial features. Subsequently, to enhance feature representations, these local features are processed by the Transformer-based Feature Interaction and Fusion module, which leverages Cross-Covariance Attention (CCA) to reduce computational complexity while facilitating the exchange and fusion of feature tokens from various perspectives. Extensive experiments showed that CT-MIFNet demonstrated the superior performance and generalization ability on the BCI Competition IV-2a, BCI Competition IV-2b, and the EEG datasets for laser-evoked pain datasets, achieving accuracies of 81.67%, 86.75%, and 83.48%, respectively. To enhance model interpretability, t-distributed stochastic neighbor embedding (t-SNE) and heatmap were employed for visualization. The code is available at https://github.com/XCZchaos/CT-MIFNet.git.}
}
```


## **Contact**
For questions or issues, please open an issue or contact:
<a href="mailto:asherxiong552@gmail.com">📧 asherxiong552@gmail.com</a>
