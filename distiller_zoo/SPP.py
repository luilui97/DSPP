#coding=utf-8
from helper.dynamic_loss import MultiLossLayer
import math
import torch
import torch.nn.functional as F

import torch.nn as nn
# create SPP layers
class SPP(torch.nn.Module):

    def __init__(self, num_levels=5, pool_type='avg_pool'):
        super(SPP, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten

class SPPloss(torch.nn.Module):

    def __init__(self,max_weight,min_weight,only_last,spp_layer,topk):
        super(SPPloss, self).__init__()
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.only_last = only_last
        self.spp_layer = spp_layer
        self.topk = topk


    def forward(self, g_s,g_t):

        if self.only_last:
            s = []
            t = []
            s.append(g_s)
            t.append(g_t)
            for (index,data) in enumerate(zip(s,t)):
                if index == (len(s)-1):  #if is not the last feature,use spp loss

                    return(self.spp_depcoupled_loss(data[0],data[1]))
        else:
            loss = []
            for (index,data) in enumerate(zip(g_s,g_t)):
                if index != (len(g_s)-1):  #if is not the last feature,use spp loss
                    loss.append(self.spp_loss(data[0],data[1]))
                else:
                    loss.append(self.spp_depcoupled_loss(data[0],data[1]))
            return loss


    def spp_loss(self,f_s,f_t):
        spp = SPP(num_levels=self.spp_layer)
        g_s = spp(f_s)
        g_t = spp(f_t)
        batch = g_s.shape[0]
        for i in range(batch):
            norma_s = F.normalize(g_s[i],dim=0)
            norma_t = F.normalize(g_t[i],dim=0)
            if i == 0:
                loss = torch.dist(norma_s,norma_t,p=2)
            else:
                loss = loss + torch.dist(norma_s,norma_t,p=2)
        return(loss/batch)

    def spp_depcoupled_loss(self,f_s,f_t):
        spp = SPP(num_levels=self.spp_layer)
        g_s = spp(f_s)
        g_t = spp(f_t)
        batch = g_s.shape[0]
        for i in range(batch):
            norma_s = F.normalize(g_s[i],dim=0)
            norma_t = F.normalize(g_t[i],dim=0)
            if i == 0:
                loss = self.depcoupled_dist(norma_s,norma_t)
            else:
                loss = loss + self.depcoupled_dist(norma_s,norma_t)
        return(loss/batch)

    def depcoupled_dist(self,norma_s,norma_t):
        max_weight = self.max_weight
        min_weight = self.min_weight

        vec_lenth = norma_t.shape[0]
        top_k = int(vec_lenth/self.topk)
        max_values,max_index = torch.topk(norma_t,top_k)
        min_values,min_index = torch.topk(norma_t,(vec_lenth-top_k),largest=False)

        dist_max = torch.dist(norma_t[max_index],norma_s[max_index],p=2)
        dist_min = torch.dist(norma_t[min_index],norma_s[min_index],p=2)
        return max_weight*dist_max+min_weight*dist_min


