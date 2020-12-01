# LOB Modeling

### Introduction

The current project page provides pytorch code that implements the following paper:   
**Title:**      "Benchmark Dataset for Short-Term Market Prediction of Limit Order Book in China Markets"    
**Authors:**     Charles Huang, Weifeng Ge, Hongsong Chou, Xin Du

**Institution:** FinAI Laboratory, Hong Kong Graduate School of Advanced Studies     
**Code:**        https://github.com/hkgsas/LOB/tree/master/lob_modeling  
**Paper:**       https://github.com/hkgsas/LOB/blob/master/Benchmark%20Dataset%20for%20Short-Term%20Market%20Prediction%20of%20Limit%20Order%20Book%20in%20China%20Markets%202020%20Nov%20v3.pdf

**Abstract:**
Limit Order Book (LOB) has generated “big financial data” for analysis and prediction from both academic community and industry practitioners.  This paper presents a benchmark LOB dataset of China stock market, covering a few thousand stocks for the period of June to September 2020.  Experiment protocols are designed for model performance evaluation: at the end of every second, to forecast the upcoming volume-weighted average price (VWAP) change and volume over 12 horizons ranging from 1 second to 300 seconds. Results based on linear regression model and state-of-the-art deep learning models are compared. Practical short-term trading strategy framework based on the alpha signal generated is presented. 

### Platform
This code was developed and tested with pytorch version 1.7.0

### Setting

You can download lob dataset from [here](https://github.com/hkgsas/LOB).

Download the dataset, and put them in the path 
'tt.arg.dataset_root/'

In ```train.py```, replace the dataset root directory with your own:
tt.arg.dataset_root = ''



### Training

```
# ************************** linear regression *****************************
$ CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset linear_regression --device 0 2>&1 | tee loss.log

# ************************ multilayer perceptron ***************************
$ CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset mlp --device 0 2>&1 | tee loss.log

# ******************************** LSTM ************************************
$ CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset lstm --device 0 2>&1 | tee loss.log

# ********************************* CNN ************************************
$ CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset cnn --device 0 2>&1 | tee loss.log

# ****************************** CNN-LSTM **********************************
$ CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset cnn_lstm --device 0 2>&1 | tee loss.log

```

### Other Parameters
Please set these parameters accordingly.

    #tt.arg.model = 'cnn_lstm' if tt.arg.model is None else tt.arg.model
    #tt.arg.input_size = 124 if tt.arg.input_size is None else tt.arg.input_size
    #tt.arg.class_num  = 5 if tt.arg.class_num is None else tt.arg.class_num 
    #tt.arg.epoch = 20 if tt.arg.epoch is None else tt.arg.epoch    
    #tt.arg.timestep = 3 if tt.arg.timestep is None else tt.arg.timestep
    #tt.arg.sequence_len = 50 if tt.arg.sequence_len is None else tt.arg.sequence_len
    #tt.arg.model_type = None if tt.arg.model_type is None else tt.arg.model_type
