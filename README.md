# LOB
## Benchmark Dataset of Limit Order Book in China Markets

FinAI Laboratory

Hong Kong Graduate School of Advanced Studies

contact@gsas.edu.hk

### Table of Contents
0. [Introduction](#introduction)
0. [Abstract](#abstract)
0. [Pipeline](#pipeline)
0. [Codes and Installation](#codes-and-installation)
0. [Models](#models)
0. [Results](#results)

### Introduction

This repository contains the dataset and [codes](https://github.com/hkgsas/LOB/tree/master/lob_modeling) described in the paper ["Benchmark Dataset for Short-Term Market Prediction of Limit Order Book in China Markets"](https://github.com/hkgsas/LOB/blob/master/Benchmark%20Dataset%20for%20Short-Term%20Market%20Prediction%20of%20Limit%20Order%20Book%20in%20China%20Markets%202020%20Nov%20v3.pdf). Five baseline models, inculding linear regression (LR), multilayer perceptron (MLP), convolutional neural network (CNN), long short term memory (LSTM), and CNN-LSTM, are tested on the proposed benchmark dataset.

**Note**

0. All algorithms are implemented based on the deep learning framework [PyTorch](https://pytorch.org/).
0. Our PyTorch version is 1.7.0. If you are in a lower version, please modify the codes accordingly.

### Abstract

Limit Order Book (LOB) has generated “big financial data” for analysis and prediction from both academic community and industry practitioners. This paper presents a benchmark LOB dataset of China stock market, covering a few thousand stocks for the period of June to September 2020. Experiment protocols are designed for model performance evaluation: at the end of every second, to forecast the upcoming volume-weighted average price (VWAP) change and volume over 12 horizons ranging from 1 second to 300 seconds. Results based on linear regression model and state-of-the-art deep learning models are compared. Practical short-term trading strategy framework based on the alpha signal generated is presented.


### Pipeline
0. Pipeline of the proposed selective joint fine-tuning:
	![Selective Joint Fine-tuning Pipeline](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/blob/master/selective_joint_ft/cvpr2017_img1.png)


### Codes and Installation
0. Add new layers into Caffe:
	- [caffe.proto](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/blob/master/selective_joint_ft/additional_layers/caffe_proto_additional.txt)
	- [MergeData](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/blob/master/selective_joint_ft/additional_layers/merge_data_layer.hpp)
	- [SplitData](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/blob/master/selective_joint_ft/additional_layers/split_data_layer.hpp)
	- [RandomCropBoostedData](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/blob/master/selective_joint_ft/additional_layers/random_crop_boosted_data_layer.hpp)
	- [FeatureStatistics](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/blob/master/selective_joint_ft/additional_layers/feature_statistics_layer.hpp)
	- [NormalKnnMatch](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/blob/master/selective_joint_ft/additional_layers/normal_knn_match_layer.hpp)
	- [RefinedHistFeature](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/blob/master/selective_joint_ft/additional_layers/refined_hist_feature_layer.hpp)
	- [Residual](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/blob/master/selective_joint_ft/additional_layers/residual_layer.hpp)

0. Image Retrieval:
	- [get the features statistics](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/tree/master/selective_joint_ft/image_retrieval/feature_stats)
	- [extract histogram features](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/tree/master/selective_joint_ft/image_retrieval/feature_extraction)
	- [online nearest neighbor searching](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/tree/master/selective_joint_ft/image_retrieval/knn_searching)
	
0. Selective Joint Fine-tuning:
	- [joint fine-tuning](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/tree/master/selective_joint_ft/joint_training)

### Models

0. Visualizations of network structures (tools from [ethereon](http://ethereon.github.io/netscope/quickstart.html)):
	- [Selective Joint Fine-tuning: ResNet-152] (http://ethereon.github.io/netscope/#/gist/8bdda026e3391eacfa43cc24f4f4a9ff)

0. Model files:
	- ResNet 152 Net pretrained on ImageNet ILSVRC 2012: [deploy](https://gist.github.com/ZYYSzj/c5c80129db55594238f16c10a3f8e108),  [model](https://drive.google.com/drive/folders/0B3sl2RWJv33ZRTNyRkdkNzBnRVU).
	- ResNet 152 Net pretrained on Places 205: [deploy](https://gist.github.com/ZYYSzj/c5c80129db55594238f16c10a3f8e108), [model](https://drive.google.com/drive/folders/0B3sl2RWJv33ZRTNyRkdkNzBnRVU).
	- Stanford Dogs 120: [deploy](https://gist.github.com/ZYYSzj/8bdda026e3391eacfa43cc24f4f4a9ff), [model](https://drive.google.com/drive/folders/0B3sl2RWJv33ZR1ZkazQxWEZ2TGc).
	- Oxford Flowers 102: [deploy](https://gist.github.com/ZYYSzj/8bdda026e3391eacfa43cc24f4f4a9ff), [model](https://drive.google.com/drive/folders/0B3sl2RWJv33ZdGp4dXk2RUpXNG8).
	- Caltech 256: [deploy](https://gist.github.com/ZYYSzj/8bdda026e3391eacfa43cc24f4f4a9ff), [model](https://drive.google.com/drive/folders/0B3sl2RWJv33ZSjZSSFVmQ3R0TDA).
	- Mit Indoor 67: [deploy](https://gist.github.com/ZYYSzj/8bdda026e3391eacfa43cc24f4f4a9ff), [model](https://drive.google.com/drive/folders/0B3sl2RWJv33ZWUFwNk1wMnZhaTg).

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
