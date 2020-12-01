from torchtools import *
from collections import OrderedDict
import math
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from lib.model import StructMatchNet, FeatureCorrelation, featureL2Norm
#import glog as log
from lib.cbam import CBAM
import glog as log
import _init_paths
from tools.hrnet import HRNet
from wide_resnet import WideResnet

def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(int(h_pad), int(w_pad)))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp
    
def feature_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    feature_pyramid = []
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2
        #maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(int(h_pad), int(w_pad)))
        maxpool = nn.AvgPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(int(h_pad), int(w_pad)))
        x = maxpool(previous_conv)
        feature_pyramid.append(x)
    return feature_pyramid
    
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# class ASPP(nn.Module):
    # def __init__(self, backbone, output_stride, BatchNorm):
        # super(ASPP, self).__init__()
        # if backbone == 'drn':
            # inplanes = 512
        # elif backbone == 'mobilenet':
            # inplanes = 320
        # else:
            # inplanes = 2048
        # if output_stride == 16:
            # dilations = [1, 6, 12, 18]
        # elif output_stride == 8:
            # dilations = [1, 12, 24, 36]
        # else:
            # raise NotImplementedError

        # self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        # self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        # self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        # self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             # nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             # BatchNorm(256),
                                             # nn.ReLU())
        # self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        # self.bn1 = BatchNorm(256)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        # self._init_weight()

    # def forward(self, x):
        # x1 = self.aspp1(x)
        # x2 = self.aspp2(x)
        # x3 = self.aspp3(x)
        # x4 = self.aspp4(x)
        # x5 = self.global_avg_pool(x)
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        # x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        # return self.dropout(x)

    # def _init_weight(self):
        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
                # # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # # m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()


# def build_aspp(backbone, output_stride, BatchNorm):
    # return ASPP(backbone, output_stride, BatchNorm)
    
class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 8, 16, 32]
        elif output_stride == 8:
            dilations = [1, 4, 8, 16]
        elif output_stride == 4:
            dilations = [1, 2, 4, 8]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 64, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 64, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 64, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 64, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 64, 1, stride=1, bias=False),
                                             BatchNorm(64),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(320, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, userelu=True, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False))

        if tt.arg.normtype == 'batch':
            self.layers.add_module('Norm', nn.BatchNorm2d(out_planes, momentum=momentum, affine=affine, track_running_stats=track_running_stats))
        elif tt.arg.normtype == 'instance':
            self.layers.add_module('Norm', nn.InstanceNorm2d(out_planes))

        if userelu:
            self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        self.layers.add_module(
            'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
        return out

class ConvNet(nn.Module):
    def __init__(self, opt, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvNet, self).__init__()
        self.in_planes  = opt['in_planes']
        self.out_planes = opt['out_planes']
        self.num_stages = opt['num_stages']
        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list and len(self.out_planes)==self.num_stages)

        num_planes = [self.in_planes,] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else True

        conv_blocks = []
        for i in range(self.num_stages):
            if i == (self.num_stages-1):
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1], userelu=userelu))
            else:
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1]))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0),-1)
        return out



# encoder for imagenet dataset
class EmbeddingImagenet(nn.Module):
    def __init__(self,
                 emb_size):
        super(EmbeddingImagenet, self).__init__()
        # set size
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                              out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        output_data = self.conv_4(self.conv_3(self.conv_2(self.conv_1(input_data))))
        return self.layer_last(output_data.view(output_data.size(0), -1))

# encoder for imagenet dataset
class AsppEmbeddingImagenet(nn.Module):
    def __init__(self,
                 emb_size):
        super(AsppEmbeddingImagenet, self).__init__()
        # set size
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        # self.conv_4 = ASPP(inplanes=self.hidden*2, output_stride=4, BatchNorm=nn.BatchNorm2d)
        # self.drop_4 = nn.Dropout2d(0.5)
        # self.conv_5 = ASPP(inplanes=self.hidden*4, output_stride=4, BatchNorm=nn.BatchNorm2d)
        # self.drop_5 = nn.Dropout2d(0.5)
        # self.pool_6 = nn.AdaptiveAvgPool2d((1, 1))
        # self.layer_last = nn.Sequential(nn.Linear(in_features=self.hidden*4,
                                              # out_features=self.emb_size, bias=True),
                                        # nn.BatchNorm1d(self.emb_size))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.conv_5 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*4,
                                              out_channels=self.hidden*8,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 8),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.conv_6 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*8,
                                              out_channels=self.hidden*16,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 16),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.drop_7 = nn.Dropout2d(0.5)
        self.layer_last = nn.Sequential(nn.Linear(in_features=21504,
                                              out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        conv_feat   = self.conv_6(self.conv_5(self.conv_4(self.conv_3(self.conv_2(self.conv_1(input_data))))))
        output_data = spatial_pyramid_pool(conv_feat,int(conv_feat.size(0)),[int(conv_feat.size(2)),int(conv_feat.size(3))],[1,2,4])
        output_data = self.drop_7(output_data)      
        return self.layer_last(output_data.view(output_data.size(0), -1))
        


class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 3,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)

        # get eye matrix (batch_size x 2 x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).to(node_feat.device)

        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)

        # compute attention and aggregate
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1), node_feat)

        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)

        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        return node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 2, 1, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                       out_channels=self.num_features_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

        if self.separate_dissimilarity:
            # layers
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                           out_channels=self.num_features_list[l],
                                                           kernel_size=1,
                                                           bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                                )
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1)
            self.dsim_network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # compute abs(x_i, x_j)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = F.sigmoid(self.sim_network(x_ij))

        if self.separate_dissimilarity:
            dsim_val = F.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val


        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).to(node_feat.device)
        edge_feat = edge_feat * diag_mask
        merge_sum = torch.sum(edge_feat, -1, True)
        # set diagonal as zero and normalize
        edge_feat = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1) * merge_sum
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0), torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)), 0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).to(node_feat.device)
        edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)

        return edge_feat


class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 dropout=0.0):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout

        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, node_feat, edge_feat):
        # for each layer
        edge_feat_list = []
        for l in range(self.num_layers):
            # (1) edge to node
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)

            # (2) node to edge
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)

            # save edge feature
            edge_feat_list.append(edge_feat)

        # if tt.arg.visualization:
        #     for l in range(self.num_layers):
        #         ax = sns.heatmap(tt.nvar(edge_feat_list[l][0, 0, :, :]), xticklabels=False, yticklabels=False, linewidth=0.1,  cmap="coolwarm",  cbar=False, square=True)
        #         ax.get_figure().savefig('./visualization/edge_feat_layer{}.png'.format(l))


        return edge_feat_list

# encoder for imagenet dataset
class FeaturePyramidImagenet(nn.Module):
    def __init__(self,
                 emb_size):
        super(FeaturePyramidImagenet, self).__init__()
        # set size
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))


    def forward(self, input_data):
        conv_feat   = self.conv_4(self.conv_3(self.conv_2(self.conv_1(input_data))))
        output_data = feature_pyramid_pool(conv_feat,int(conv_feat.size(0)),[int(conv_feat.size(2)),int(conv_feat.size(3))],[1,4,7])
        #output_data.append(conv_feat)
        return output_data

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

class FeaturePyramidHRnet(nn.Module):
    def __init__(self, cfg, pretrained, cuda_device):
        super(FeaturePyramidHRnet, self).__init__()
        # set size
        self.stem_net = WideResnet(depth=28, widen_factor=10, drop_rate=0.0,
                                   pool="none", extra_block=False,
                                   block_strides=[2,2,2,2], 
                                   extra_block_width_mult=1,
                                   num_layers=[2,2,2,2],)
        self.FeatureCorrelation = FeatureCorrelation(shape='4D',normalization=False)       
        self.color_channels     = 3 

    def forward(self, input_data):
        if self.training:
            conv_feat   = self.stem_net(input_data)
            output_data = feature_pyramid_pool(conv_feat,int(conv_feat.size(0)),[int(conv_feat.size(2)),int(conv_feat.size(3))],[1,5])
            output_data[1] = output_data[0].repeat(1,1,5,5) 
               
            return output_data
        else:
            n, c, h, w    = input_data.shape        
            grid_size     = int(math.sqrt(c / self.color_channels))
            reshaped_data = input_data.view(n * grid_size * grid_size, self.color_channels, h, w)                  
            conv_feat     = self.stem_net(reshaped_data)
            output_data   = F.avg_pool2d(conv_feat, kernel_size=conv_feat.size()[2:])
            output_data   = output_data.squeeze(-1).squeeze(-1)
            n, c          = output_data.shape
            output_data   = output_data.view(-1,grid_size*grid_size,c).permute(0,2,1).contiguous()               
            conv_feat     = output_data.view(-1,c,grid_size,grid_size)

            output_data    = feature_pyramid_pool(conv_feat,int(conv_feat.size(0)),[int(conv_feat.size(2)),int(conv_feat.size(3))],[1,grid_size])
            output_data[1] = output_data[0].repeat(1,1,grid_size,grid_size) 

            return output_data                                            

class StructNodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[1, 1],
                 dropout=0.0):
        super(StructNodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [int(num_features * r) for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 3,
                groups=1 if l > 0 else num_features,
                out_channels=self.num_features_list[l],
                kernel_size=3,
                padding=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l])
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()
            layer_list['cbam{}'.format(l)] = CBAM(gate_channels=self.num_features_list[l])

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat, spatial_att):
        # get size
        num_tasks = node_feat.size(0)
        num_data  = node_feat.size(1)
        n,d,c,h,w = node_feat.size()
        spatial_att = spatial_att.detach()
        #print("==========================Node Update=============================") 
        #node_feat = node_feat.view(num_tasks,num_data,-1)
        #print(node_feat.shape,node_feat)
        # get eye matrix (batch_size x 2 x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).to(node_feat.device)

        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)
        edge_feat = edge_feat.view(n,2,d,d,1,1).repeat(1,1,1,1,h,w)
        spatial_att = spatial_att.view(n,1,d,d,h,w).repeat(1,2,1,1,1,1)
        spatial_att = edge_feat * spatial_att
        spatial_att = spatial_att.sum(dim=3)
        spatial_att = spatial_att.view(n,2,d,1,h,w).repeat(1,1,1,c,1,1)
        sim_att     = spatial_att[:,0,:,:,:,:]
        dsim_att    = spatial_att[:,1,:,:,:,:]
        node_sim    = sim_att  * node_feat
        node_dsim   = dsim_att * node_feat         
        # compute attention and aggregate
        aggr_feat = torch.cat([node_feat,node_sim,node_dsim],dim=3)
        node_feat = aggr_feat.view(n*d,c*3,h,w)
        #print(spatial_att.shape,node_feat.shape,node_sim.shape,node_dsim.shape)   
        # non-linear transform
        node_feat = self.network(node_feat)
        
        #params_dict = dict(self.network.named_parameters())
        #for key, v in params_dict.items():
        #    print(key,v)
                
        node_feat = node_feat.view(n,d,node_feat.size(1),node_feat.size(2),node_feat.size(3))
        #print(node_feat.shape,node_feat)
        #print("==========================<            >==========================")
        #assert 0==1 
        return node_feat

class StructEdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 2, 1, 1],
                 dropout=0.0):
        super(StructEdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        self.sim_network = StructMatchNet(ncons_kernel_sizes = [3,3,3],ncons_channels=[4,4,1])

    def forward(self, node_feat, edge_feat):
        # compute abs(x_i, x_j)
        # sim_val = batch_size x 1 x node_size x node_size
        sim_val, spatial_att = self.sim_network(node_feat)
        sim_val  = sim_val.unsqueeze(1)
        dsim_val = 1.0 - sim_val
        #print("============Similarity Value===========")
        #print(sim_val.shape,sim_val[0])
        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).to(node_feat.device)
        edge_feat = edge_feat * diag_mask
        merge_sum = torch.sum(edge_feat, -1, True)
        #print(merge_sum.shape,merge_sum[0])
        # set diagonal as zero and normalize
        #print(edge_feat.shape,edge_feat[0])
        aggr_feat = torch.cat([sim_val, dsim_val], 1) * edge_feat
        #print(aggr_feat.shape,aggr_feat[0]) 
        edge_feat = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1) * merge_sum
        #print(edge_feat.shape,edge_feat[0])
        edge_feat[:,:,:,-1] = edge_feat[:,:,-1,:]
        #print(edge_feat.shape,edge_feat[0])
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0), torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)), 0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).to(node_feat.device)
        edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)
        #print(edge_feat.shape,edge_feat[0])
        return edge_feat, spatial_att

class StructGraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 dropout=0.0):
        super(StructGraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.debug_mode = False
        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = StructNodeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            # set node to edge
            node2edge_net = StructEdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, node_feats, edge_feats):
        # for each layer
        edge_feat_lists = []
        for h in range(1,len(node_feats)):
            #print(node_feats[h].shape)
            #print("===============================")
            edge_feat_list = []
            for l in range(self.num_layers):
                # (1) edge to node
                if l == 0:
                    node_feat = node_feats[h]
                    edge_feat = edge_feats
                    #edge_feat, spatial_att = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)
                #if self.debug_mode:
                #    log.info("Node Update.")
                #node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat, spatial_att)
                #if self.debug_mode:
                #    log.info("Edge Update.")
                # (2) node to edge
                edge_feat, spatial_att = self._modules['node2edge_net{}'.format(l)](node_feats[h], edge_feat)
                
                # save edge feature
                edge_feat_list.append(edge_feat)
            edge_feat_lists.append(edge_feat_list) 
        
        edge_feat_lists = list(map(list, zip(*edge_feat_lists)))
        edge_feat_lists = [torch.mean(torch.stack(edge_feat_list), dim=0) for edge_feat_list in edge_feat_lists]

        # if tt.arg.visualization:
        #     for l in range(self.num_layers):
        #         ax = sns.heatmap(tt.nvar(edge_feat_list[l][0, 0, :, :]), xticklabels=False, yticklabels=False, linewidth=0.1,  cmap="coolwarm",  cbar=False, square=True)
        #         ax.get_figure().savefig('./visualization/edge_feat_layer{}.png'.format(l))


        return edge_feat_lists


class LinearRegression(nn.Module):
    def __init__(self, in_features, class_num):
        super(LinearRegression, self).__init__()
        # set size
        self.embedding_net = nn.Linear(in_features=in_features, out_features=class_num, bias=True)

    def forward(self, x):
        y = self.embedding_net(x)
        return y 

class MLP(nn.Module):
    def __init__(self, in_features, class_num):
        super(MLP, self).__init__()
        # set size
        self.layer_1 = nn.Sequential(nn.Linear(in_features=in_features, out_features=512, bias=True),
                                    nn.ReLU(inplace=True))
        self.layer_2 = nn.Sequential(nn.Linear(in_features=512, out_features=1024, bias=True),
                                    nn.ReLU(inplace=True))
        self.layer_3 = nn.Sequential(nn.Linear(in_features=1024, out_features=1024, bias=True),
                                    nn.ReLU(inplace=True))
        self.layer_4 = nn.Sequential(nn.Linear(in_features=1024, out_features=64, bias=True),
                                    nn.ReLU(inplace=True))
        self.layer_5 = nn.Sequential(nn.Linear(in_features=64, out_features=class_num, bias=True))

    def forward(self, x):
        y = self.layer_5(self.layer_4(self.layer_3(self.layer_2(self.layer_1(x)))))
        return y 

'''
class ConvNet(nn.Module):
    def __init__(self, in_features, class_num):
        super(ConvNet, self).__init__()
        # set size
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=[3,25],
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=64),
                                    nn.MaxPool2d(kernel_size=[2,2]),
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                              out_channels=128,
                                              kernel_size=[3,5],
                                              padding=1,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[1,2]),
                                    nn.BatchNorm2d(num_features=128),
                                    nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=128,
                                              out_channels=256,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[1,2]),
                                    nn.BatchNorm2d(num_features=256),
                                    nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=256,
                                              out_channels=512,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[1,2]),
                                    nn.BatchNorm2d(num_features=512),
                                    nn.ReLU(inplace=True))
        self.layer_5 = nn.Sequential(nn.Linear(in_features=512, out_features=class_num, bias=True)) 

    def forward(self, x):
        conv_feat = self.conv_1(x)
        #print(conv_feat.shape)            
        conv_feat = self.conv_2(conv_feat)
        #print(conv_feat.shape) 
        conv_feat = self.conv_3(conv_feat)
        #print(conv_feat.shape) 
        conv_feat = self.conv_4(conv_feat)
        #print(conv_feat.shape)  
        conv_feat = F.avg_pool2d(conv_feat, kernel_size=conv_feat.size()[2:])
        conv_feat = conv_feat.squeeze(-1).squeeze(-1)
        #print(conv_feat.shape) 
        #assert 0==1  
        y = self.layer_5(conv_feat)
        return y
'''

class ConvNet(nn.Module):
    def __init__(self, in_features, class_num):
        super(ConvNet, self).__init__()
        # set size
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=[4,124],
                                              padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=64),
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                              out_channels=128,
                                              kernel_size=[1,1],
                                              padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=128),
                                    nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=128,
                                              out_channels=256,
                                              kernel_size=[4,1],
                                              padding=0,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[2,1]),
                                    nn.BatchNorm2d(num_features=256),
                                    nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=256,
                                              out_channels=512,
                                              kernel_size=[3,1],
                                              padding=0,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[2,1]),
                                    nn.BatchNorm2d(num_features=512),
                                    nn.ReLU(inplace=True))
        self.layer_5 = nn.Sequential(nn.Linear(in_features=512, out_features=class_num, bias=True)) 

    def forward(self, x):
        conv_feat = self.conv_1(x)
        #print(conv_feat.shape)            
        conv_feat = self.conv_2(conv_feat)
        #print(conv_feat.shape) 
        conv_feat = self.conv_3(conv_feat)
        #print(conv_feat.shape) 
        conv_feat = self.conv_4(conv_feat)
        #print(conv_feat.shape)  
        conv_feat = F.avg_pool2d(conv_feat, kernel_size=conv_feat.size()[2:])
        conv_feat = conv_feat.squeeze(-1).squeeze(-1)
        #print(conv_feat.shape) 
        #assert 0==1  
        y = self.layer_5(conv_feat)
        return y

class LstmNet(nn.Module):
    def __init__(self, in_features, seq_len, class_num):
        super(LstmNet, self).__init__()
        # set size
        self.in_features = in_features
        self.seq_len     = seq_len 
        self.class_num = class_num         
        self.rnn = nn.LSTM(in_features, in_features, num_layers=1)
        self.relu= nn.ReLU(inplace=False) 
        self.classifier = nn.Sequential(nn.Linear(in_features=in_features*seq_len, out_features=class_num, bias=True)) 

    def forward(self, x):
        x_reshaped = x.permute(1,0,2)
        batch_size = x_reshaped.size()[1]          
        h0 = torch.zeros(1, batch_size, self.in_features).to(x.device)
        c0 = torch.zeros(1, batch_size, self.in_features).to(x.device)
        #print(x_reshaped.shape,x_reshaped)        
        output, (hn,cn) = self.rnn(x_reshaped,(h0,c0))
        output = output.permute(1,0,2).contiguous()
        b,l,c = output.shape
        output = output.view(b,l*c)        
        y = self.classifier(self.relu(output))
        return y


class ConvLstmNet(nn.Module):
    def __init__(self, in_features, seq_len, class_num):
        super(ConvLstmNet, self).__init__()
        # set size
        self.in_features = in_features
        self.seq_len     = seq_len 
        self.class_num = class_num 

        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=[4,124],
                                              padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=64),
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                              out_channels=128,
                                              kernel_size=[1,1],
                                              padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=128),
                                    nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=128,
                                              out_channels=256,
                                              kernel_size=[4,1],
                                              padding=0,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[2,1]),
                                    nn.BatchNorm2d(num_features=256),
                                    nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=256,
                                              out_channels=512,
                                              kernel_size=[3,1],
                                              padding=0,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[2,1]),
                                    nn.BatchNorm2d(num_features=512),
                                    nn.ReLU(inplace=True))
        self.rnn = nn.LSTM(512, 64, num_layers=1)
        self.relu= nn.ReLU(inplace=False)         
        self.layer_5 = nn.Sequential(nn.Linear(in_features=640, out_features=class_num, bias=True)) 

    def forward(self, x):
        #print(x.shape) 
        conv_feat = self.conv_1(x)
        #print(conv_feat.shape)            
        conv_feat = self.conv_2(conv_feat)
        #print(conv_feat.shape) 
        conv_feat = self.conv_3(conv_feat)
        #print(conv_feat.shape) 
        conv_feat = self.conv_4(conv_feat)
        #print(conv_feat.shape)  
        conv_feat = conv_feat.squeeze(-1).permute(2,0,1)
        #print(conv_feat.shape) 
        rnn_feat,_  = self.rnn(conv_feat)
        #print(rnn_feat.shape) 
        rnn_feat  = rnn_feat.permute(1,0,2).contiguous()
        #print(rnn_feat.shape) 
        b,l,d     = rnn_feat.shape
        rnn_feat  = rnn_feat.view(b,l*d)
        #print(rnn_feat.shape) 
        rnn_feat  = self.relu(rnn_feat)
        #print(rnn_feat.shape)    
        y = self.layer_5(rnn_feat)
        #print(y.shape)
        #assert 0==1  
        return y