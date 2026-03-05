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

## **Project Structure**
```
CT-MIFNet/  
|—— prcessing/                                
    └── BCI_2a_getData.m                    # Script for extraction and preprocessing of 2a dataset  
    └── BCI_2b_getData.m                    # Script for extraction and preprocessing of 2b dataset  
|—— commom_spatial_pattern.py               # CSP function of the 2a dataset  
|—— commom_spatial_pattern_2b.py            # CSP function of the 2b dataset  
|—— commom_spatial_pattern_newdata.py       # CSP function of the EDLEP dataset 
|—— config.py                               # Global configuration file for the 2a dataset project  
|—— config_2b.py                            # Global configuration file for the 2b dataset project  
|—— config_newdata.py                       # Global configuration file for the EDLEP dataset project  
|—— CT_MIFNet.py                            # Including CT-MIFNet and model training framework applied to 2a dataset  
|—— CT_MIFNet_2b.py                         # Including CT-MIFNet and model training framework applied to 2b dataset  
|—— CT_MIFNet_newdata.py                    # Including CT-MIFNet and model training framework applied to EDLEP dataset 
|—— Figure_01.png                           # CT-MIFNetm model framework diagram  
|—— Figure_04.png                           # Comparison chart of results between CT-MIFNet and other models on three datasets  
|—— LICENSE                                 # License file  
|—— metric.py                               # Including custom functions such as FFT and LabelSmoothingLoss  
|—— metric_newdata.py                               # Including custom functions for EDLEP dataset such as FFT and LabelSmoothingLoss  

```
---

## **File Descriptions**

### 1.prcessing file  
- Execute ```BCI_2a_getData.m``` in Matlab to extract and preprocess the 2a dataset.
- Execute ```BCI_2a_getData_2b.m``` in Matlab to extract and preprocess the 2b dataset.

### 2.```commom_spatial_pattern.py``` , ```commom_spatial_pattern_2b.py``` and ```commom_spatial_pattern_newdata.py```
- Script for implementing multi-class CSP.
  
### 3. ```config.py``` , ```config_2b.py``` and ```config_newdata.py```
A centralized file for managing global configurations, including:
- Parameter settings for the dataset (e.g. number of channels, time points, sampling frequency).
- Model hyperparameters (e.g., embedding dimensions, attention heads).
- Training parameters (e.g., batch size, epochs, learning rate，num_class).

### 4.```CT_MIFNet.py``` , ```CT_MIFNet_2b.py```and ```CT_MIFNet_newdata.py```
Contains three main modules: ```CT_MIFNet()```, ```Trans()``` and ```main()```.
- ```CT_MIFNet()``` is the core implementation of the model.
- ```Trans()``` implements training and evaluation logic.
- ```Main()``` is used to configure the environment (random seed), create instances of the Trans() class, and perform training for each subject.

---

## **Supported Datasets**
CT_MIFNet supports two EEG datasets and one Pain Percept dataset. Below are the recommended settings:  
### 1.BCI Competition IV dataset 2a
- Model Initialization：```CT_MIFNet()```
- Key Parameters:```classes=4```
### 2.BCI Competition IV dataset 2b  
- Model Initialization：```CT_MIFNet()```
- Key Parameters: ```classes=2```
### 3.EDLEP dataset
- The data will be made public soon. Please stay tuned. When the data is made public, we will update the description of this section.

---


## **Usage**

You can directly run the model files CT_MIFNet.py , CT_MIFNet_2b.py and CT_MIFNet_newdata. We will update the descriptions for some files shortly after our paper is accepted.

---

## Notice⚠️⚠️⚠️
1.If you apply CT-MIFNet to other paradigms besides Motor Imagery, we recommend that you remove the CSP module. Based on our experiments in the Pain Perception paradigm, we found that the CSP had a negative impact on the model.


2.Dataset III has been made public. we had released the code of CT-MIFNet on Dataset III.

---

# News🎉🎉🎉
Dataset III is now available at https://openneuro.org/datasets/ds005285/versions/1.0.0 and we have uploaded our code. we selected subjects 1, 3, 5, 8, 10, 12, 15, 18 and 28 in our experiments.


---

## Acknowledges
This code is based on the following repository. We would like to express our gratitude to these authors for their contributions to open-source work.

<a href="https://github.com/eeyhsong/EEG-Transformer.git">https://github.com/eeyhsong/EEG-Transformer.git

<a href="https://github.com/lixujin1999/TFF-Former.git">https://github.com/lixujin1999/TFF-Former.git

---

## **Citation**
Hope this code can be useful. I would appreciate you citing us in your paper. 💐😊
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

If you use Dataset III in your research, please cite the following paper. 😁
```
@article{zhao2025comprehensive,
  title={A comprehensive EEG dataset of laser-evoked potentials for pain research},
  author={Zhao, Xiangyue and Zhou, Jingyao and Zhang, Libo and Zhuang, Yun and Duan, Haoqing and Wei, Shiyu and Yao, Suchen and Lu, Xuejing and Bi, Yanzhi and Hu, Li},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={1536},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
---

## **Contact**
For questions or issues, please open an issue or contact:
<a href="mailto:asherxiong552@gmail.com">📧 asherxiong552@gmail.com</a>
