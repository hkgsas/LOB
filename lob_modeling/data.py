from __future__ import print_function
from torchtools import *
import torch.utils.data as data
import random
import os
import numpy as np
from PIL import Image as pil_image
from PIL import Image
import pickle
from itertools import islice
from torchvision import transforms
import glog as log
import cv2

import csv
import math
import pandas as pd
from io import StringIO
from zipfile import ZipFile
import copy

class SequentialDataLoader(data.Dataset):
    def __init__(self, root, path, data_dim, label_dim, timesetp,    
            partition='Train',
            ratio=1.0, 
            normalization=True,
            quantize=True,      
            model_type='linear', 
            sequence_len=10):
        self.root = root
        self.path = root + '/' + path
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.timesetp = timesetp 
        self.partition = partition
        self.ratio = ratio
        self.normalization = normalization
        self.quantize = quantize          
        self.model_type = model_type
        self.sequence_len = sequence_len
        # label assignment
        self.class_num = 5

        super(SequentialDataLoader, self).__init__()
        
        # read dataset list
        self.path_list  = []
        with open(self.path) as csvfile: 
            csv_reader = csv.reader(csvfile) 
            for row in csv_reader:
                self.path_list.append(row[0])

        self.x_list = []
        self.y_list = []
        self.z_list = []  
        # prepare data
        for path in self.path_list: 
            # set dataset information 
            path = self.root + '/' + path
            log.info(path)         
            data = self.data_reader(path)
            data_len = data.shape[0]
            data_dim = data.shape[1]        
            assert data_len > 1
            assert self.data_dim + 2*(self.label_dim+1) == data_dim
            log.info(path)
            x = data[:, :self.data_dim]
        
            if self.normalization:
                x = self.data_normalization(x)

            y = data[:, (self.data_dim+1):(self.data_dim+1+self.label_dim)]
            avg_price = data[:,0]
            sigma     = data[:,self.data_dim]
            #print(avg_price,sigma) 
            avg_price = np.expand_dims(avg_price, axis=1)
            sigma     = np.expand_dims(sigma, axis=1)
            y = y / avg_price
            y = np.log(y) / sigma 
 
            y = self.label_assignment(y,path)

            x_list, y_list, z_list = self.data_formulation(x,y)
            self.x_list[-1:-1] = x_list
            self.y_list[-1:-1] = y_list
            self.z_list[-1:-1] = z_list             

        self.data_len = len(self.x_list)

        log.info("Current mode %s, data length %d",self.partition,self.data_len)

    def data_reader(self,path):
        csv_data  = []
        with open(path) as csvfile: 
            csv_reader = csv.reader(csvfile) 
            for row in csv_reader:
                row_data = []
                idx = 0
                for row_item in row:
                    if idx >= 2:
                        row_items = row_item.split('|')
                        for element in row_items:
                            row_data.append(float(element))
                    idx = idx + 1
                row_data = np.array(row_data)
                csv_data.append(row_data) 
        csv_data = np.array(csv_data)
        return csv_data 

    def data_normalization(self, x):
        mean = np.mean(x,axis=0,keepdims=True)
        std  = np.std(x,axis=0,keepdims=True)         
        x    = (x - mean)/(std+1e-20)
        return x         

    def quantile_p(self, data, p):
        #print(data.max(),data.min())
        data = sorted(data,reverse=True)                                 
        pos = (len(data) + 1) * p
        if pos >= len(data):
            pos = len(data) - 1    
        pos_integer = int(math.modf(pos)[1])
        pos_decimal = pos - pos_integer
        Q = data[pos_integer - 1] + (data[pos_integer] - data[pos_integer - 1])*pos_decimal
        return Q

    def label_assignment(self, y, path):
        # remove noise
        percentage = [0.9, 0.7, 0.3, 0.1]         
        current_y = copy.deepcopy(y)
        # normalization
        n, dim = current_y.shape

        for d in range(0, dim):
            quantile = [] 
            for i in range(0, len(percentage)):
               Q = self.quantile_p(current_y[:,d], percentage[i])
               quantile.append(Q)
            #print(current_y[:,d],quantile)
            #assert 0==1 
            y_copy = copy.deepcopy(current_y[:,d])
            current_y[y_copy < quantile[0], d] = 0
            current_y[y_copy >= quantile[-1], d] = len(percentage)             
            for i in range(0, len(percentage)-1):
                upper_bound = quantile[i+1]
                lower_bound = quantile[i]
                index1 = y_copy >= lower_bound
                index2 = y_copy < upper_bound
                index  = index1 & index2   
                current_y[index, d] = i + 1
    
        transfored_y = current_y[:,self.timesetp]
        '''
        n,d = current_y.shape
        statistics = np.zeros((5,12),dtype='int32')
        for k in range(0,n):
            for m in range(0,12):
                idx = int(current_y[k][m])
                statistics[idx][m] = statistics[idx][m] + 1
                #print(k,m,idx,self.trainval_y[k][m],statistics[idx][m])
        print(path,statistics)
        #assert 0==1
        '''
        return transfored_y

    def data_formulation(self,x,y):
        weights = [1.0,0.3,0.1,0.3,1.0]        
        x_len, x_dim = x.shape
        y_len        = y.shape 
        x_list = []
        y_list = []
        z_list = []
        for i in range(0, x_len-self.sequence_len):                 
            x_frame = x[i:i+self.sequence_len]
            y_frame = y[i+self.sequence_len]
            z_frame = weights[int(y_frame)]
            x_list.append(x_frame)
            y_list.append(y_frame)
            z_list.append(z_frame) 
        return x_list,y_list,z_list

    def get_data_element(self,index):
        data_element_x = self.x_list[index]
        data_element_y = self.y_list[index]
        data_element_z = self.z_list[index]        
        data_element_x = torch.from_numpy(np.asarray(data_element_x)).float()
        data_element_y = torch.from_numpy(np.asarray(data_element_y)).float()
        data_element_z = torch.from_numpy(np.asarray(data_element_z)).float()
        if self.model_type == 'linear':
            data_element_x = data_element_x.view(-1)
            data_element_y = data_element_y.view(-1)
            data_element_z = data_element_z.view(-1)            
        elif self.model_type == 'conv':
            data_element_x = data_element_x.unsqueeze(0)
            data_element_y = data_element_y.view(-1)
            data_element_z = data_element_z.view(-1)            
        elif self.model_type == 'lstm':
            data_element_x = data_element_x
            data_element_y = data_element_y.view(-1)
            data_element_z = data_element_z.view(-1)
        else:
            assert 0==1,'Unkown model type'

        return data_element_x, data_element_y, data_element_z

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        return self.get_data_element(index)
