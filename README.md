# LOB
## Benchmark Dataset of Limit Order Book in China Markets

FinAI Laboratory

Hong Kong Graduate School of Advanced Studies

contact@gsas.edu.hk

### Table of Contents
0. [Introduction](#introduction)
0. [Abstract](#abstract)
0. [Keywords](#keywords)
0. [Models](#models)
0. [Results](#results)

### Introduction

This repository contains the dataset and [codes](https://github.com/hkgsas/LOB/tree/master/lob_modeling) described in the paper ["Benchmark Dataset for Short-Term Market Prediction of Limit Order Book in China Markets"](https://github.com/hkgsas/LOB/blob/master/Benchmark%20Dataset%20for%20Short-Term%20Market%20Prediction%20of%20Limit%20Order%20Book%20in%20China%20Markets%202020%20Nov%20v3.pdf). Five baseline models, inculding linear regression (LR), multilayer perceptron (MLP), convolutional neural network (CNN), long short term memory (LSTM), and CNN-LSTM, are tested on the proposed benchmark dataset.

**Note**

0. All algorithms are implemented based on the deep learning framework [PyTorch](https://pytorch.org/).
0. Our PyTorch version is 1.7.0. If you are in a lower version, please modify the codes accordingly.

### Abstract

Limit Order Book (LOB) has generated “big financial data” for analysis and prediction from both academic community and industry practitioners. This paper presents a benchmark LOB dataset of China stock market, covering a few thousand stocks for the period of June to September 2020. Experiment protocols are designed for model performance evaluation: at the end of every second, to forecast the upcoming volume-weighted average price (VWAP) change and volume over 12 horizons ranging from 1 second to 300 seconds. Results based on linear regression model and state-of-the-art deep learning models are compared. Practical short-term trading strategy framework based on the alpha signal generated is presented.

### Keywords 
High-Frequency Trading, Limit Order Book, Artificial Intelligence, Machine Learning, Deep Neural Network, Short-Term Price Prediction, Alpha Signal, Trading Strategies, China Stock Market

### Models
0. Configuration of the linear regression model:
	![Linear Regression](https://github.com/hkgsas/LOB/blob/master/lr.png)

0. Configuration of the multilayer perceptron model:
	![Multilayer Perceptron](https://github.com/hkgsas/LOB/blob/master/mlp.png)
	
0. Configuration of the shallow LSTM model:
	![Long Short Term Memory](https://github.com/hkgsas/LOB/blob/master/mlp.png)
	
0. Configuration of the CNN model:
	![Convolutional Neural Network](https://github.com/hkgsas/LOB/blob/master/cnn.png)

0. Configuration of the CNN-LSTM model:
	![CNN-LSTM](https://github.com/hkgsas/LOB/blob/master/cnnlstm.png)

### Installation and Usage
Please refer to the [ReadMe.txt](https://github.com/hkgsas/LOB/blob/master/lob_modeling/README.md) in ./lob_modeling to install and run experiments.

### Results

0. Multi crop testing accuracy on Stanford Dogs 120 (in the same manner with that in [VGG-net](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)):

	Method|mean Accuracy(%)
	:---:|:---:
	[HAR-CNN](http://www.linyq.com/hyper-cvpr2015.pdf)|49.4
	[Local Alignment](https://link.springer.com/article/10.1007/s11263-014-0741-5)|57.0
	[Multi Scale Metric Learning](https://arxiv.org/abs/1402.0453)|70.3
	[MagNet](https://arxiv.org/abs/1511.05939)|75.1
	[Web Data + Original Data](https://arxiv.org/abs/1511.06789)|85.9
	Target Only Training from Scratch|53.8
	Selective Joint Training from Scratch|83.4
	Fine-tuning w/o source domain|80.4
	Selective Joint FT with all source samples|85.6
	Selective Joint FT with random source samples|85.5
	Selective Joint FT w/o iterative NN retrieval|88.3
	Selective Joint FT with Gabor filter bank|87.5
	Selective Joint FT|90.2
	Selective Joint FT with Model Fusion|90.3
	
0. Multi crop testing accuracy on Oxford Flowers 102 (in the same manner with that in [VGG-net](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)):

	Method|mean Accuracy(%)
	:---:|:---:
	[MPP](http://ieeexplore.ieee.org/document/7301274/)|91.3
	[Multi-model Feature Concat](https://arxiv.org/abs/1406.5774)|91.3
	[MagNet](https://arxiv.org/abs/1511.05939)|91.4
	[VGG-19 + GoogleNet + AlexNet](https://arxiv.org/abs/1506.02565)|94.5
	Target Only Training from Scratch|58.2
	Selective Joint Training from Scratch|80.6
	Fine-tuning w/o source domain|90.2
	Selective Joint FT with all source samples|93.4
	Selective Joint FT with random source samples|93.2
	Selective Joint FT w/o iterative NN retrieval|94.2
	Selective Joint FT with Gabor filter bank|93.8
	Selective Joint FT|94.7
	Selective Joint FT with Model Fusion|95.8
	[VGG-19 + Part Constellation Model](https://arxiv.org/abs/1504.08289)|95.3
	Selective Joint FT with val set|97.0

0. Multi crop testing accuracy on Caltech 256 (in the same manner with that in [VGG-net](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)):

	Method|mean Acc(%) 15/class|mean Acc(%) 30/class|mean Acc(%) 45/class|mean Acc(%) 60/class
	:---:|:---:|:---:|:---:|:---:
	[M-HMP](http://rse-lab.cs.washington.edu/papers/multipath-sparse-coding-cvpr-13.pdf)|40.5 ± 0.4|48.0 ± 0.2|51.9 ± 0.2|55.2 ± 0.3
	[Z.&F. Net](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)|65.7 ± 0.2|70.6 ± 0.2|72.7 ± 0.4|74.2 ± 0.3
	[VGG-19](https://arxiv.org/abs/1409.1556)|-|-|-|85.1 ± 0.3
	[VGG-19 + GoogleNet + AlexNet](https://arxiv.org/abs/1506.02565)|-|-|-|86.1
	[VGG-19 + VGG-16](https://arxiv.org/abs/1409.1556)|-|-|-|86.2 ± 0.3
	Fine-tuning w/o source domain|76.4 ± 0.1|81.2 ± 0.2|83.5 ± 0.2|86.4 ± 0.3
	Selective Joint FT|80.5 ± 0.3|83.8 ± 0.5|87.0 ± 0.1|89.1 ± 0.2

0. Multi crop testing accuracy on MIT Indoor 67 (in the same manner with that in [VGG-net](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)):

	Method|mean Accuracy(%)
	:---:|:---:
	[MetaObject-CNN](https://arxiv.org/abs/1510.01440)|78.9
	[MPP + DFSL](https://pdfs.semanticscholar.org/33d6/a99d497540a17783d237013483dbfa506cd7.pdf)|80.8
	[VGG-19 + FV](https://www.robots.ox.ac.uk/~vgg/publications/2015/Cimpoi15a/cimpoi15a.pdf)|81.0
	[VGG-19 + GoogleNet](https://arxiv.org/abs/1506.02565)|84.7
	[Multi Scale + Multi Model Ensemble](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Herranz_Scene_Recognition_With_CVPR_2016_paper.pdf)|86.0
	Fine-tuning w/o source domain|81.7
	Selective Joint FT with ImageNet|82.8
	Selective Joint FT with Places|85.8
	Selective Joint FT with hybrid data|85.5
	Average the output of Places and hybrid data|86.9
