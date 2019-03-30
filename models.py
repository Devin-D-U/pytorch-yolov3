from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from PIL import Image

from utils.parse_config import *
from utils.utils import build_targets
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_modules(modules_defs):
    '''
    Constructs module list of layer blocks from
    module configuration in modules_defs
    '''
    hyperparams = modules_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i modules_def in enumerate(modules_defs):
        modules=nn.Sequential()

        if module_def["type"] =="convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size-1)//2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels = output_filters[-1],
                    out_channels = filters,
                    kernel_size = kernel_size,
                    stride = int(module_def["stride"]),
                    padding = pad,
                    bias = not bn.
                ).
            )
            if bn:
                modules.add_module("batch_norm_%d" %i,nn.BatchNorm2d(filters))
            if module_def["activation"]=="leaky":
                modules.add_module("leaky_%d" %i,nn.LeakyReLU(0.1))
        elif module_def["type"]=="maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size ==2 and stride ==1:
                padding =nn.ZeroPad2d((0,1,0,1))
                modules.add_module("_debug_padding_%d" % i,padding)
            maxpool = nn.MaxPool2d(
                kernel_size = int(module_def["size"]),
                stride = int(module_def["stride"]),
                padding = int(kernel_size - 1 //2),
            )
            modules.add_module("maxpool_%d" % i ,maxpool)
        elif module_def["type"]=="upsample":
            upsample = nn.Upsample(scale_factor = int(module_def["stride"]),mode="nearest")
            moduels.add_module("upsample_%d" % i ,upsample)
        elif module_def["type"]=="route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module("route_%d" % i ,EmptyLayer())
        elif module_def["type"]=="shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i ,EmptyLayer)
        elif module_def["type"] =="yolo":
            anchor_idxs [int(x) for x in module_def["mask"].split(",")]
            #extract anchors
            anchors = [int(x) for x in module_def["anchors"].split[',']]
            anchors = [(anchors[i],anchor[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            #Define detection layer
            yolo_layer = YOLOLayer(anchors,num_classes,img_height)
            modules.add_module("yolo_%d" % i ,yolo_layer)
        # register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)
    return hyperparams,module_list

class EmptyLayer(nn.Module):
    '''
    placeholder for route and shortcut layers
    '''
    def __init__(self):
        super(EmptyLayer,self).__init__()

class YOLOLayer(object):
    """docstring for YOLOLayer. detection layer """

    def __init__(self, anchors,num_classes,img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.bbox_attrs = 5+num_classes
        self.image_dim =img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True) # coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True) # confidence loss
        self.ce_loss = nn.CrossEntropyLoss() # class loss

    def forward(self,x,targets = None):
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride =self.image_dim /nG

        #tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuad.ByteTensor id x.is_cuda else torch.ByteTensor

        prediction = x.view(nB,nA,self.bbox_attrs,nG,nG).permute(0,1,3,4,2).contiguous()
        
