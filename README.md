# LOB
## Benchmark Dataset of Limit Order Book in China Markets

FinAI Laboratory

Hong Kong Graduate School of Advanced Studies

contact@gsas.edu.hk

### Table of Contents
0. [Introduction](#introduction)
1. [Abstract](#abstract)
2. [Keywords](#keywords)
3. [Models](#models)
4. [Data Format](#data)
5. [Installation and Usage](#install)
6. [Results](#results)

### Introduction

This repository contains the [dataset](https://drive.google.com/file/d/13xgOAXhVa1QhZLg4DaqWwbCZVNhXNKiw/view?usp=sharing) and [codes](https://github.com/hkgsas/LOB/tree/master/lob_modeling) described in the paper ["Benchmark Dataset for Short-Term Market Prediction of Limit Order Book in China Markets"](https://github.com/hkgsas/LOB/blob/master/Benchmark%20Dataset%20for%20Short-Term%20Market%20Prediction%20of%20Limit%20Order%20Book%20in%20China%20Markets%202020%20Nov%20v3.pdf). Five baseline models, inculding linear regression (LR), multilayer perceptron (MLP), convolutional neural network (CNN), long short term memory (LSTM), and CNN-LSTM, are tested on the proposed benchmark dataset.

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

### Data Format
The folder structure of the LOB [dataset](https://drive.google.com/file/d/13xgOAXhVa1QhZLg4DaqWwbCZVNhXNKiw/view?usp=sharing) is like the following.
```
   .\LOB_data
         .\2020.6
	 .\2020.7
	 .\2020.8
	 .\2020.9
	 lob_sz_6789_train_val.txt
	 lob_sz_678_train.txt
	 lob_sz_9_val.txt 
```
"lob_sz_678_train.txt" is the file list used to train the machine learning models, and "lob_sz_9_val.txt" is the file list used to test the accuracy as the validation. In each folder under ".\LOB_data", there are monthly LOB data in ".csv" format for many different stocks.



### Installation and Usage
Please refer to the [ReadMe.txt](https://github.com/hkgsas/LOB/blob/master/lob_modeling/README.md) in ./lob_modeling to install and run experiments.


### Results

0. Model performance metrics for different horizons computed on the test folds
        ![Results](https://github.com/hkgsas/LOB/blob/master/results.png)
