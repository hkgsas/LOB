import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import time, argparse
import json, sys
# 
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

#work-around to allow import from directory on same level
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from scipy.stats import linregress
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

from torchtools import *
from data import SequentialDataLoader
from model import FeaturePyramidHRnet, AsppEmbeddingImagenet, GraphNetwork, ConvNet, FeaturePyramidImagenet, StructGraphNetwork
from model import LinearRegression, MLP, ConvNet, LstmNet, ConvLstmNet 
from model import feature_pyramid_pool
from lib.model import StructMatchNet, FeatureCorrelation, featureL2Norm
import shutil
import os
import random
#import nvidia_smi
#import seaborn as sns
import glog as log
import _init_paths
from tools.hrnet import HRNet
import pickle
from losses import OnlineTripletLoss
from sampler import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from rotate_utils import create_rotations_labels,create_4rotations_images
from datetime import datetime
from utils.Functions import *
from utils.LabelSmoothing import LSR

'''
def gpu_info(device_ids=None):
    nvidia_smi.nvmlInit()
    for device_id in device_ids:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        #print(f'id: {device_id}, gpu: {res.gpu}%, gpu-mem: {res.memory}%')
'''

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes, dtype='uint8')[y]

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class NormalizedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormalizedConv, self).__init__()
        self.conv_ = nn.Conv2d(in_channels = in_channels, \
                               out_channels = out_channels, \
                               kernel_size=1,\
                               padding=0, \
                               bias=False)
        self.conv_ = self.conv_.cuda()        
    
    def forward(self, x):
        # copy weight
        # self.conv_.state_dict()['weight'] = featureL2Norm(self.conv_.state_dict()['weight'])
        # get results
        y = self.conv_(x)
        n,c,h,w = y.shape
        y = y.view(n,c)
        return y

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ModelTrainer(object):
    def __init__(self,
                 model,
                 data_loader,
                 dataset):
        # set encoder and gnn
        self.model = torch.nn.DataParallel(model, device_ids=tt.arg.device).cuda()
        cuda_device = 'cuda:'+str(self.model.device_ids[0])
        self.model.to(cuda_device)

        # get data loader
        self.data_loader = data_loader
        self.dataset     = dataset

        # tensorboard log directory
        time_now      = datetime.now().isoformat()
        self.log_path = os.path.join(tt.arg.log_root, time_now)                                
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(log_dir=self.log_path)        
        
        # Xavier initialization
        print(self.model)    
        
        # warmstart
        self.warmup_step     = tt.arg.warmup_iteration         
        self.warmstart_flag  = True        

        # set optimizer
        params_dict = dict(self.model.named_parameters())
        base_lr = tt.arg.lr
        weight_decay = tt.arg.weight_decay
        params = []
        for key, v in params_dict.items():
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*1.00, 'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*1.00, 'weight_decay': weight_decay*0, 'name': key}]

        self.module_params = params
        
        # set optimizer
        self.optimizer = optim.SGD(params=self.module_params,
                                    lr=tt.arg.lr,
                                    momentum=tt.arg.momentum,
                                    weight_decay=tt.arg.weight_decay)

        # set loss
        #self.weights     = [1.0,1.0,1.0,1.0,1.0]
        self.weights     = [1.0,0.3,0.1,0.3,1.0]
        self.weights     = torch.from_numpy(np.asarray(self.weights)).float().cuda()  
        #self.criterion   = torch.nn.CrossEntropyLoss(weight=self.weights).cuda()
        self.criterion   = torch.nn.CrossEntropyLoss().cuda()

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0
        self.debug_mode = False

        # resume training
        if tt.arg.resume:
            model_state_file = 'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar'
            if os.path.isfile(model_state_file):
                checkpoint       = torch.load(model_state_file,map_location=cuda_device)
                self.global_step = checkpoint['iteration']
                self.val_acc     = checkpoint['val_acc']
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint (Iteration {})".format(self.global_step))
            else:
                print("Training from scratch...")                  

    def train(self):
        val_acc = self.val_acc

        for epoch in range(0, tt.arg.epoch):
            self.model.train()
            for i, (data, target, weight) in enumerate(self.data_loader['train']):
                # init grad
                self.optimizer.zero_grad()
                data   = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True).long().contiguous().view(-1)  
                score  = self.model(data) 
                loss   = self.criterion(score, target)                     
                loss.backward()
                self.optimizer.step()
 
            # adjust learning rate
            if epoch % tt.arg.dec_lr == 0 and epoch != 0:
                self.adjust_learning_rate(optimizers=[self.optimizer],
                                          lr=tt.arg.lr,
                                          iter=epoch)

            # evaluation
            train_acc = self.eval(epoch,partition='train')            
            val_acc = self.eval(epoch,partition='val')
            log.info("Step %d Loss %f Train accuracy %f Validataion accuracy %f", epoch, loss, train_acc, val_acc)

            is_best = 0

            if val_acc >= self.val_acc:
                self.val_acc = val_acc
                is_best = 1

            self.save_checkpoint({
                'iteration': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'val_acc': val_acc,
                'optimizer': self.optimizer.state_dict(),
                }, is_best)

            #tt.log_step(global_step=self.global_step)

    def eval(self, epoch, partition='test', log_flag=True):
        best_acc = 0

        self.model.eval()
        valy_pred = []
        valy_test = [] 
        valy      = []                
        for i, (data, target, weight) in enumerate(self.data_loader[partition]):
            data   = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True).long().contiguous().view(-1)  
            score  = self.model(data)
            score  = nn.functional.softmax(score, dim=1)   

            valy_pred.append(score.detach())
            valy_test.append(target.detach())

            target = self.one_hot_encode(tt.arg.class_num,target) 
            valy.append(target.detach())
        
        valy_pred = torch.cat(valy_pred,dim=0).detach().cpu().numpy()
        valy_test = torch.cat(valy_test,dim=0).detach().cpu().numpy() 
        valy      = torch.cat(valy,dim=0).detach().cpu().numpy()

        # How well have we done on test data
        print("===============================================")
        print("The %d-th epoch",epoch)
        print(partition)
        Y=np.argmax(valy,axis=1)
        Yhat=np.argmax(valy_pred,axis=1)
        c=confusion_matrix(Y,Yhat)
        c=np.concatenate((c,np.sum(c,axis=1).reshape(-1,1)),axis=1)
        c=np.concatenate((c,np.sum(c,axis=0).reshape(1,-1)),axis=0)
        print(c)
        print(classification_report(Yhat,Y))
        print("cohen kappa score:",cohen_kappa_score(Yhat,Y))

        return cohen_kappa_score(Yhat,Y)


    def adjust_learning_rate(self, optimizers, lr, iter):
        ratio = 0.1 #** (int(iter / tt.arg.dec_lr))
        log.info('Adjust Learning Rate')
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * ratio
                log.info('%s: %s' % (param_group['name'], param_group['lr']))                

    def label2edge(self, label):
        # get size
        num_samples = label.size(1)

        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)

        # compute edge
        edge = torch.eq(label_i, label_j).float().to(tt.arg.device[0])

        # expand
        edge = edge.unsqueeze(1)
        edge = torch.cat([edge, 1 - edge], 1)
        return edge

    def hit(self, logit, label):
        pred = logit.max(1)[1]
        hit = torch.eq(pred, label).float()
        return hit

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes)[class_idx].to(tt.arg.device[0])

    def save_checkpoint(self, state, is_best):
        torch.save(state, 'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar')
        if is_best:
            shutil.copyfile('asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar',
                            'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'model_best.pth.tar')

def set_exp_name():
    exp_name = 'D-{}'.format(tt.arg.dataset)
    exp_name += '_N-{}_K-{}_U-{}'.format(tt.arg.num_ways, tt.arg.num_shots, tt.arg.num_unlabeled)
    exp_name += '_L-{}_B-{}'.format(tt.arg.num_layers, tt.arg.meta_batch_size)
    exp_name += '_T-{}'.format(tt.arg.transductive)
    exp_name += '_SEED-{}'.format(tt.arg.seed)

    return exp_name

if __name__ == '__main__':
    # set gpu device
    gpu_list = "'"
    device_id = []
    gpu_start_id = tt.arg.device[0]
    for gpu_id in tt.arg.device:
        gpu_list += str(gpu_id) + ","
        device_id.append(gpu_id -  gpu_start_id)
    gpu_list += "'"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    tt.arg.device = device_id 
    import glog as log   
    log.info(gpu_list,device_id)
    log.info(device_id)

    tt.arg.device = tt.arg.device if torch.cuda.is_available() else 'cpu'
    # replace dataset_root with your own
    tt.arg.dataset_root = '/data/wfge/'
    tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 1 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.num_unlabeled = 0 if tt.arg.num_unlabeled is None else tt.arg.num_unlabeled
    tt.arg.num_layers = 4 if tt.arg.num_layers is None else tt.arg.num_layers
    tt.arg.meta_batch_size  = 1750 if tt.arg.meta_batch_size is None else tt.arg.meta_batch_size
    tt.arg.epoch_iteration  = 2000 
    tt.arg.warmup_iteration = 10000    
    tt.arg.transductive = False if tt.arg.transductive is None else tt.arg.transductive
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1 if tt.arg.num_gpus is None else tt.arg.num_gpus
    tt.arg.num_workers = 3 if tt.arg.num_workers is None else tt.arg.num_workers
    tt.arg.resume      = True if tt.arg.resume is None else tt.arg.resume
    tt.arg.triplet_margin = 0.2
    tt.arg.triplet_wegith = 0.1
    tt.arg.log_root       = './log/'
    tt.arg.image_size     = 80        

    tt.log('Fewshot Learning: Dataset = %s, Setting = %d-way-%d-shot' % 
    	            (tt.arg.dataset, tt.arg.num_ways, tt.arg.num_shots))

    tt.arg.num_ways_train = tt.arg.num_ways
    tt.arg.num_ways_test = tt.arg.num_ways

    tt.arg.num_shots_train = tt.arg.num_shots
    tt.arg.num_shots_test = tt.arg.num_shots

    tt.arg.train_transductive = tt.arg.transductive
    tt.arg.test_transductive = tt.arg.transductive

    # model parameter related
    tt.arg.num_edge_features = 96
    tt.arg.num_node_features = 64
    tt.arg.emb_size = 512

    # train, test parameters
    tt.arg.train_iteration = 100000 if tt.arg.dataset == 'mini' else 60000
    tt.arg.test_iteration = 1000 #10000
    tt.arg.test_interval = 2000 if tt.arg.test_interval is None else tt.arg.test_interval
    tt.arg.test_batch_size = 1
    tt.arg.log_step = 100 if tt.arg.log_step is None else tt.arg.log_step

    tt.arg.lr = 0.01
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-4
    tt.arg.momentum = 0.9    
    tt.arg.dec_lr = 100 if tt.arg.dataset == 'mini' else 20000
    tt.arg.dropout = 0.1 if tt.arg.dataset == 'mini' else 0.0

    tt.arg.experiment = set_exp_name() if tt.arg.experiment is None else tt.arg.experiment

    print(set_exp_name())

    #set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tt.arg.log_dir_user = tt.arg.log_dir if tt.arg.log_dir_user is None else tt.arg.log_dir_user
    tt.arg.log_dir = tt.arg.log_dir_user

    if not os.path.exists('asset/checkpoints'):
        os.makedirs('asset/checkpoints')
    if not os.path.exists('asset/checkpoints/' + tt.arg.experiment):
        os.makedirs('asset/checkpoints/' + tt.arg.experiment)

    tt.arg.model = 'cnn_lstm' if tt.arg.model is None else tt.arg.model
    # create model
    tt.arg.input_size = 124 if tt.arg.input_size is None else tt.arg.input_size
    tt.arg.class_num  = 5 if tt.arg.class_num is None else tt.arg.class_num 
    tt.arg.epoch = 20 if tt.arg.epoch is None else tt.arg.epoch    
    tt.arg.timestep = 3 if tt.arg.timestep is None else tt.arg.timestep
    tt.arg.sequence_len = 50 if tt.arg.sequence_len is None else tt.arg.sequence_len
    tt.arg.model_type = None if tt.arg.model_type is None else tt.arg.model_type
    if tt.arg.model == 'linear_regression': 
        model = LinearRegression(tt.arg.input_size*tt.arg.sequence_len, tt.arg.class_num)
        tt.arg.model_type = 'linear'
    elif tt.arg.model == 'mlp':      
        model = MLP(tt.arg.input_size*tt.arg.sequence_len, tt.arg.class_num)
        tt.arg.model_type = 'linear'
    elif tt.arg.model == 'cnn': 
        model = ConvNet(tt.arg.input_size, tt.arg.class_num)
        tt.arg.model_type = 'conv'
    elif tt.arg.model == 'lstm':       
        model = LstmNet(tt.arg.input_size, tt.arg.sequence_len, tt.arg.class_num)
        tt.arg.model_type = 'lstm'
    elif tt.arg.model == 'cnn_lstm':
        model = ConvLstmNet(tt.arg.input_size, tt.arg.sequence_len, tt.arg.class_num)
        tt.arg.model_type = 'conv'
    else:
        raise Exception("Invalid model type!")        


    train_dataset= SequentialDataLoader(root='/data/wfge/ai_projects/dataset/LOB_data/', 
                                      path='lob_sz_678_train.txt', data_dim=124, 
                                      label_dim=12, timesetp=tt.arg.timestep,    
                                      partition='Train',
                                      ratio=0.7, 
                                      normalization=True,
                                      quantize=True,      
                                      model_type=tt.arg.model_type, 
                                      sequence_len=tt.arg.sequence_len)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                      batch_size=tt.arg.meta_batch_size*tt.arg.num_gpus,
                                      shuffle=True,
                                      num_workers=tt.arg.num_workers,
                                      pin_memory=True)
    valid_dataset= SequentialDataLoader(root='/data/wfge/ai_projects/dataset/LOB_data/', 
                                      path='lob_sz_9_val.txt', data_dim=124, 
                                      label_dim=12, timesetp=tt.arg.timestep,     
                                      partition='Validataion',
                                      ratio=0.7, 
                                      normalization=True,
                                      quantize=True,      
                                      model_type=tt.arg.model_type, 
                                      sequence_len=tt.arg.sequence_len)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                      batch_size=tt.arg.meta_batch_size*tt.arg.num_gpus,
                                      shuffle=True,
                                      num_workers=tt.arg.num_workers,
                                      pin_memory=True)

    # load data
    data_loader = {'train': train_loader,
                   'val':   valid_loader}

    tt.arg.config      = './net/HRNet-Image-Classification/experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
    tt.arg.pretrained  = '' #'./net/HRNet-Image-Classification/output/imagenet/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100/checkpoint.pth.tar'
    tt.arg.cuda_device = 'cuda:'+str(0)
    
    '''
    tt.arg.weights = '../BF3S/experiments/tieredImageNet/WRNd28w10CosineClassifierRotAugRotSelfsupervision/feature_extractor_net_epoch52.best'  
    checkpoint     = torch.load(tt.arg.weights,map_location=tt.arg.cuda_device)

    pretrained_dict= checkpoint['network'] 
    enc_model_dict = enc_module.state_dict()   

    updated_dic    = {}
    for k,v in pretrained_dict.items():
        key = "stem_net." + k
        if key in enc_model_dict:
            updated_dic[key] = v

    for k, _ in updated_dic.items():
        log.info('=> loading {} pretrained model {}'.format(k, tt.arg.weights))
            
    enc_model_dict.update(updated_dic)
    enc_module.load_state_dict(updated_dic)
    '''

    # create trainer
    trainer = ModelTrainer(model=model,
                           data_loader=data_loader,
                           dataset=train_dataset)

    trainer.train()
