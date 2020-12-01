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

0. Model performance metrics for different horizons computed on the test folds

	Method|mean Acc(%) 15/class|mean Acc(%) 30/class|mean Acc(%) 45/class|mean Acc(%) 60/class
	:---:|:---:|:---:|:---:|:---:
	[M-HMP](http://rse-lab.cs.washington.edu/papers/multipath-sparse-coding-cvpr-13.pdf)|40.5 ± 0.4|48.0 ± 0.2|51.9 ± 0.2|55.2 ± 0.3
	[Z.&F. Net](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)|65.7 ± 0.2|70.6 ± 0.2|72.7 ± 0.4|74.2 ± 0.3
	[VGG-19](https://arxiv.org/abs/1409.1556)|-|-|-|85.1 ± 0.3
	[VGG-19 + GoogleNet + AlexNet](https://arxiv.org/abs/1506.02565)|-|-|-|86.1
	[VGG-19 + VGG-16](https://arxiv.org/abs/1409.1556)|-|-|-|86.2 ± 0.3
	Fine-tuning w/o source domain|76.4 ± 0.1|81.2 ± 0.2|83.5 ± 0.2|86.4 ± 0.3
	Selective Joint FT|80.5 ± 0.3|83.8 ± 0.5|87.0 ± 0.1|89.1 ± 0.2

