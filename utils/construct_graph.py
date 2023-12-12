from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import repeat
import numpy as np
import logging
# 准备数据
data = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]

class Graph_fy():
    def __init__(self, args, knownclasses):
    # 创建一个空的无向图
        self.G = nx.Graph()
        args["n_neighbors"] = 10
        self.n_neighbors=args["n_neighbors"]
        self.knownclasses=knownclasses
        self.increment = args["increment"]
        self.alpha = args["lp_alpha"]
        self.rw_alpha = args["rw_alpha"]
        self.T = args["rw_T"]
        self.use_topo = args["use_topo"]
        

    # 计算KNN近邻
    def KNN(self, data, data_label, targets_label, n_neighbor=2):
        #k = 2
        knn = KNeighborsClassifier(n_neighbors=n_neighbor)
        #A = kneighbors_graph(data, n_neighbor, mode='distance', metric='euclidean')
        knn.fit(data_label, targets_label)
        distances, indices = knn.kneighbors(data)
        return distances, indices

    def Graph(self, data, targets, pseutargets, logits=None, pseulabel=None,  unlabel_index = None, T = 5):
        
        index_label = np.where(unlabel_index==1)[0]
        
        data_label = data[index_label]
        targets_label = targets[index_label]
        #A = self.KNN(data, n_neighbor=self.n_neighbors)
        distances, indices = self.KNN(data, data_label, targets_label, n_neighbor=self.n_neighbors)

        
        # 将数据点添加为图的节点
        for i, point in enumerate(data):
            self.G.add_node(i, attr=point)

        # 添加边到图中
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:
                self.G.add_edge(i, neighbor, weight=distances[i][1])
        
            
        ## 生成邻接矩阵
        adjcentmatrix = nx.to_numpy_matrix(self.G)
        D = adjcentmatrix.sum(1)
        adjcentmatrix_hat = adjcentmatrix/D ## 沿着列加和是1
        #pse_label_0 = self.generate_pseudo_label(data, targets, pseutargets, self.adjcentmatrix, unlabel_index, logits, pseulabel)
        
        ''' 
        ## 生成种子区域匹配矩阵
        for i in range(T):
            pi +=  self.matchingmatrix(adjcentmatrix_hat, self.rw_alpha, t=i)
        
        matching_matrix = self.rw_alpha * pi  ##［N,N］
        matching_matrix_seed = matching_matrix[:,index_label]  # [N,index_label]
        '''

    def matchingmatrix(self, P, alpha, t=2):
        matrix = (1-alpha)^t * torch.matrix_power(self.adjcentmatrix, t)
        return matrix



    def generate_pseudo_label(self, output, targets, pseutargets, logits, pseulabel, unlabel_index, alpha=0.8, module='output'):
        #matrix = adjcentmatrix
        #index_withself = torch.diag(matrix) == 1
        #matrix = matrix - torch.eye(output.shape[0], dtype=int) * index_withself #matrix是對稱的
        #self.Graph(output, targets, pseutargets, logits, pseulabel, unlabel_index, alpha=0.8, module='output')
        #adjcentmatrix = self.adjcentmatrix
        train = True
        
        if train:
            label_mask = unlabel_index
        else:
            label_mask = torch.zeros_like(targets, dtype=torch.int32)

        if 'output' in module:
        
            ### semantic statis
            num_sp, num_class = output.shape[0], output.shape[1]
            thre = 0.6
            logits = torch.tensor(logits)
            softmax = nn.Softmax(dim=1)
            logits[:,:self.knownclasses-self.increment] = -10
            soft_label = softmax(logits)#[:,self.knownclasses-self.increment:]*1 + logits[:,:self.knownclasses-self.increment]*1e-5)
            #hard_label_propa, hard_label = soft_label.max(1)
            
            #pse_mask = hard_label_propa>thre  #选择模型输出大于阈值的样本
            output = torch.tensor(output)
            normal_out = F.normalize(output)
            dis_hard = self.get_elu_dis(normal_out) #计算模型输出表示的相似度
            #dis_hard = get_elu_dis(soft_label)
            gamma=1
            simi_hard = torch.exp(- dis_hard * gamma)
            if self.use_topo==True:
                softmax_top = nn.Softmax(dim=0)
                def matchingmatrix(P, alpha, t=2):
                    matrix = ((1-alpha)**t) * torch.matrix_power(P, t)  # 
                    return matrix
                P = softmax_top(F.cosine_similarity(output.unsqueeze(1), output, dim=-1) * 10)  # 列求和是1 [N,N]
                pi = torch.zeros_like(P) #pi = [] #
                for i in range(5):
                    pi += (matchingmatrix(P, self.rw_alpha, t=(i+1)))
                #pi = torch.concatenate(pi, dim=1)  # [N,N,T]
                #pi_con_dis = pi.reshape(num_sp, -1) # [N, N*T]
                signature_vector = self.get_elu_dis(pi) # [N,N]
                simi_topo = torch.exp(- signature_vector * 10)
                simi_hard = simi_hard + simi_topo


            #### 均衡采样
            #bl_mask, bl_ind = balance_sample(output, hard_label, pse_mask_thre=pse_mask, num_sam=20) 
            #label_pselabel_bl = (label_mask+bl_mask)>0
            label_mask = torch.tensor(label_mask)
            elu_dis_0 = torch.einsum('nm, m->nm', simi_hard, label_mask)  #只选择和标记样本的相似度

        if 'output' in module: # or with_self
            index_propa = (elu_dis_0.sum(1)>0).int()  #和核心节点的相似度大于1的节点
            filter = ((elu_dis_0 - elu_dis_0.topk(40,1)[0].min(1)[0].unsqueeze(1))>=0) 
            elu_dis_0_f = elu_dis_0 * filter
            elu_dis_00 = softmax(elu_dis_0_f /0.06+torch.ones_like(elu_dis_0_f/0.06)*1e-5)  #作用是放缩，令有相似性的值变得更大，并使得加和为1
            sharp_elu_dis_0 = torch.einsum('nl,n->nl', elu_dis_00, index_propa) #和核心节点的相似度大于1的节点，在后面基本上是全部的节点
            #filter = ((sharp_elu_dis_0 - sharp_elu_dis_0.topk(40,1)[0].min(1)[0].unsqueeze(1))>=0)  #相似度从大到小前40个样本
            #sharp_elu_dis_0 = sharp_elu_dis_0*filter
            #range.append(0)
        
        L = sharp_elu_dis_0
        index_propa = torch.tensor((L.sum(1)>0).int())
        #label_pse = label_pse * pse_mask.unsqueeze(1)
        one_hotlabel = self.onehot(torch.tensor(targets), self.knownclasses) * label_mask.unsqueeze(1) 
                       #+ (1-label_mask.unsqueeze(1)) * label_pse  # 不加伪标签
        propgation = torch.einsum('vn, nd -> vd', L, one_hotlabel) #表示利用图的邻接矩阵确定一阶邻居，并将有标签的样本的标签传递给一阶邻居。
        
        #soft_label[:,:self.knownclasses] = 0
        
        pse_label = torch.einsum('n, nd -> nd', (self.alpha * index_propa + (1-index_propa)), soft_label) + \
                (1 - self.alpha) * propgation  # propgation标签一定是对的
        pse_label_0 = pse_label.clone()
        index_pseudo = torch.where(label_mask==0)[0]
        logging.info(f"pred_ratio, {(pseulabel[index_pseudo]==targets[index_pseudo]).sum()/index_pseudo.shape[0]}, \
              lp_ratio, {(pse_label.argmax(1)[index_pseudo]==torch.tensor(targets[index_pseudo])).sum()/index_pseudo.shape[0]}, \
              p_ratio, {(propgation.argmax(1)[index_pseudo]==torch.tensor(targets[index_pseudo])).sum()/index_pseudo.shape[0]}")  #4200
        return pse_label_0
        
    def onehot(self, label, numclass):
        mask = (label>=0).float()
        label_cor = torch.where(label>=0, label, torch.tensor(0))
        one_hot = torch.nn.functional.one_hot(label_cor, numclass)
        one_hot = one_hot * mask.unsqueeze(1)
        return one_hot
    
    def get_elu_dis(self, data):
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
    

'''
# 绘制图形
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True)
plt.show()
'''
