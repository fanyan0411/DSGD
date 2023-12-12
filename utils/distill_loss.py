import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from models.data import TwoStreamBatchSampler
import os, json, wandb, math
from pathlib import Path
from torch.nn import functional as F
from einops import repeat

def distill_loss(logits_old, logits, known_classes, targets, relation=None, targets_old=None):

    index_old = torch.where((targets<known_classes) * (targets>-100))[0] ## 有伪标签的旧样本
    logits = logits[index_old]
    logits_old = logits_old[index_old]
    with torch.no_grad():
        soft_logits_old = F.softmax(logits_old, dim=1)
    soft_logits_log = F.log_softmax(logits, dim=1)[:,:known_classes]
    kd_loss_soft = F.kl_div(soft_logits_log, soft_logits_old) #, reduction='mean'

    if relation=='mse':
        # 将张量标准化
        #logits_normalized = F.normalize(logits, dim=0)
        #logits_old_normalized = F.normalize(logits_old, dim=0)

        # 计算余弦相似度
        distance = torch.cdist(logits, logits)
        
        cos_sim_logits = F.cosine_similarity(logits.unsqueeze(1), logits, dim=-1)
        cos_sim_logits_old = F.cosine_similarity(logits_old.unsqueeze(1), logits_old, dim=-1)
        relat_loss = ((cos_sim_logits - cos_sim_logits_old)**2).mean()
        #print('sim', ((targets[index_old][cos_sim_logits.topk(5)[1]]-targets[index_old].unsqueeze(1))==0).sum(), 
        #      'sim_old', ((targets[index_old][cos_sim_logits_old.topk(5)[1]]-targets[index_old].unsqueeze(1))==0).sum())

    return {'kd_loss_soft': kd_loss_soft, 'relat_loss': relat_loss}

def align_loss(class_weight, increment, ancohor):
    weights = class_weight.weight
    newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
    oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
    meannew = torch.mean(newnorm)
    meanold = torch.mean(oldnorm)
    loss_align = (meannew - ancohor) ** 2 + (meanold - ancohor) **2
    ## loss_align = (meannew - ancohor) **2
    return {'loss_align': loss_align}

def matching_loss(logits_old, logits, known_classes, targets_withpse, targets, relation=None, targets_old=None, alpha=0.2, rw_alpha=0.2, T=5, gamma=1):
    index_old = torch.where((targets_withpse<known_classes) * (targets_withpse>-100))[0] ## 有伪标签的旧样本
    index_old_label = torch.where((targets[index_old]<known_classes) * (targets[index_old]>-100))[0] ## 有真实标签的旧样本
    logits = logits[index_old]
    logits_old = logits_old[index_old]
    '''
    P = attention_matrix(logits)  # [N_old,N_old]
    P_old = attention_matrix(logits_old)
     
    P = F.cosine_similarity(logits.unsqueeze(1), logits, dim=-1)
    P = P - torch.eye(P.shape[0]).to(P.device)
    P_old = F.cosine_similarity(logits_old.unsqueeze(1), logits_old, dim=-1)
    P_old = P_old -torch.eye(P_old.shape[0]).to(P_old.device)

    
    P = (P/P.sum(0).unsqueeze(0))
    P_old = (P_old/P_old.sum(0).unsqueeze(0))
    '''
    softmax = nn.Softmax(dim=0)
    P = softmax(F.cosine_similarity(logits.unsqueeze(1), logits, dim=-1) * gamma)  # 列求和是1
    P_old = softmax(F.cosine_similarity(logits_old.unsqueeze(1), logits_old, dim=-1) * gamma)
    #softmax = nn.Softmax(dim=1)
    #P = softmax(P)
    #P_old = softmax(P_old)
    
    # 计算匹配向量

    ## 生成种子区域匹配矩阵
    
    pi, pi_old = torch.zeros_like(P), torch.zeros_like(P)
    
    for i in range(T):
        pi += matchingmatrix(P, rw_alpha, t=(i+1))
        pi_old += matchingmatrix(P_old, rw_alpha, t=(i+1))
    #pi = matchingmatrix_inverse(P, rw_alpha)
    #pi_old = matchingmatrix_inverse(P_old, rw_alpha)
    '''
    pi = P #matchingmatrix(P, rw_alpha, t=1)
    pi_old = P_old #matchingmatrix(P_old, rw_alpha, t=1)
    '''
    
    matching_matrix = pi #rw_alpha * pi  ##［N,N］
    matching_matrix_old = pi_old #rw_alpha * pi_old  ##［N,N］
    
    #matching_matrix_seed = matching_matrix[:,index_old_label]  # [N,N_old_label]
    #matching_matrix_old_seed = matching_matrix_old[:,index_old_label]  ##［N,N_old_label］
    '''
    ## 计算匹配损失
    matching_loss_seed = ((matching_matrix_seed - matching_matrix_old_seed)**2).mean(1).sum()
    '''
    matching_loss_seed = ((matching_matrix - matching_matrix_old)**2).sum()
    return {'matching_loss_seed': matching_loss_seed}


def matchingmatrix(P, alpha, t=2):
    matrix = ((1-alpha)**t) * torch.matrix_power(P, t)  # 
    return matrix

def matchingmatrix_inverse(P, alpha, t=2):
    I = torch.eye(P.shape[0]).to(P.device)
    matrix = ((1-alpha)**t) * torch.inverse(I-P)  # 
    return matrix

def attention_matrix(output):
    
    normal_out = F.normalize(output)
    dis_hard = get_elu_dis(normal_out) #计算模型输出表示的相似度
    
    #dis_hard = get_elu_dis(soft_label)
    
    gamma=1
    simi_hard = torch.exp(- dis_hard * gamma)
    simi_hard = simi_hard-torch.eye(simi_hard.shape[0]).to(simi_hard.device)
    return simi_hard

def get_elu_dis(data):
        '''
        input: data [a_1, a_2, ..., a_n].T()      dim:[n, d]
        
        dist_ij = (a_i - b_j)
        '''
        n = data.shape[0]
        A2 = repeat(torch.einsum('nd -> n', data**2), 'n -> n i', i = n)
        B2 = repeat(torch.einsum('nd -> n', data**2), 'n -> i n', i = n)
        AB = data.mm(data.t())
        dist = torch.abs(A2 + B2 -2*AB)
        return torch.sqrt(dist)