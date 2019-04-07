form __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
img_path matplotlib.patches as patches

def load_classes(path):
    '''
    loads class labels at 'path'
    '''
    fp=open(path,'r')
    names = fp.read().split("\n")[:-1]
    fp.close()
    return names

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") !=-1:
        torch.nn.init_normal_(m,weight.data,0.0,0.02)
    elif classname.find("BatchNorm2d") !=-1:
        torch.nn.init_normal_(m.weight.data,1.0,0.02)
        torch.nn.init_constant_(m.bias.data,0.0)

def compute_ap(recall,precision):
    '''
    compute the average precision given the recall and precision curve
    #arguments:
        recall:list
        precision:list
    '''
    #correct Ap calculation
    mrec = np.concatenate(([0.0],recall,[1.0]))
    mpre = np.concatenate(([0.0],precision,[0.0]))

    #calculate the precision envelope
    for i in range(mpre.size-1,0,-1):
        mpre[i-1] = np.maximun(mpre[i-1],m[i])

    #to calculate area under PR curve ,look for points
    #where x axis  (recall) changes value
    i = up.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i+1] - mrec[i])* mpre[i+1])
    return ap

def bbox_iou(box1,box2,x1y1x2y2=True):
    '''
    return iou of two bounding box
    '''
    if not x1y1x2y2:
        #tansform from center and width to exact coordinate
        b1_x1,b1_x2 = box1[:,0]-box1[:,2]/2,box1[:,0]+box1[:,2]/2
        b1_y1,b1_y2 = box1[:,1]-box1[:,3]/2,box1[:,1]+box1[:,3]/2
        b2_x1,b2_x2 = box2[:,0]-box2[:,2]/2,box2[:,0]+box2[:,2]/2
        b2_y1,b2_y2 = box2[:,1]-box2[:,3]/3,box2[:,1]+box2[:,3]/2
    else:
        b1_x1,b1_y1,b1_x2,b1_y2 = box1[:,0],box1[:,1],box1[:,2],box1[:,3]
        b2_x1,b2_y1,b2_x2,b2_y2 = box2[:,0],box2[:,1],box2[:,2],box2[:,3]

    #get the coordinate of intersection rectangle
    inter_rect_x1 = torch.max(b1_x1,b2_x1)
    inter_rect_y1 = torch.max(b1_y1,b2_y1)
    inter_rect_x2 = torch.min(b1_x2,b2_x2)
    inter_rect_y2 = torch.min(b1_y2,b2_y2)

    #intersection area
    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1+1,min=0)*
            torch.clamp(inter_rect_y2-inter_rect_y1+1,min=0)

    # union area
    b1_area = (b1_x2-b1_x1+1)*(b1_y2-b1_y1+1)
    b2_area = (b2_x2-b2_x1+1)*(b2_y2-b2_y1+1)
    iou = inter_area/(b1_area+b2_area-inter_area+1e-16)
    return iou

def bbox_iou_numpy(box1,box2):
    '''
    compute iou between bounding boxes
    parameters
    box1: ndarray
    (n,4) shaped array with bboxes
    box2:ndarray
    (m,4) shaped array with bboxes
    return :ndarray
        (n,m) shaped array with contiguous
    '''
    area = (box2[:,2]-box2[:,0]) *(box2[:,3]-box[:,1])
    iw = np.minimum(np.expand_dims(box1[:,2],axis=1),box2[:,2])-
        np.maximum(np.expand_dims(box1[:,0],1),box2[:,0])
    ih = np.minimum(np.expand_dims(box1[:,3].axis=1),box2[:,3])=
        np.minimum(np.expand_dims(box1[:,1],1),box2[:,1])

    iw = np.maximum(iw,0)
    ih = np.maximum(ih,0)

    ua = np.expand_dims((box1[:,2]-box1[:,0])*(box1[:,3] - box1[:,1]),axis=1) + area -iw*ih
    ua = np.maximum(ua,np.finfo(float).eps)
    intersection = iw*ih
    return intersection/ua


def non_max_suppression(prediction,num_classes,conf_thres = 0.5.nms_thres = 0.4):
    
