import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat

def matching_loss(logits_old, logits, known_classes, targets_withpse, targets,  alpha=0.2, rw_alpha=0.2, T=5, gamma=1, targets_old = None):
    index_old = torch.where((targets_withpse<known_classes) * (targets_withpse>-100))[0] ## old samples with labels or pseudo labels
    index_old_label = torch.where((targets[index_old]<known_classes) * (targets[index_old]>-100))[0] ## old samples with labels
    #print(index_old)
    logits = logits[index_old]
    logits_old = logits_old[index_old]
    

    softmax = nn.Softmax(dim=0)  # dim = 0 not 1
    P = softmax(F.cosine_similarity(logits.unsqueeze(1), logits, dim=-1) * gamma)  # .sum(0)=1
    P_old = softmax(F.cosine_similarity(logits_old.unsqueeze(1), logits_old, dim=-1) * gamma)

    
    pi, pi_old = torch.zeros_like(P), torch.zeros_like(P)
    
    for i in range(T):
        pi += matchingmatrix(P, rw_alpha, t=(i+1))
        pi_old += matchingmatrix(P_old, rw_alpha, t=(i+1))

    
    matching_matrix = pi #rw_alpha * pi  ##［N,N］
    matching_matrix_old = pi_old #rw_alpha * pi_old  ##［N,N］
    
    #matching_matrix_seed = matching_matrix[:,index_old_label]  # [N,N_old_label]
    #matching_matrix_old_seed = matching_matrix_old[:,index_old_label]  ##［N,N_old_label］

    matching_loss_seed = ((matching_matrix - matching_matrix_old)**2).sum() #.mean(1).sum() .sum() 
    '''
    matching_matrix_sum, matching_matrix_old_sum = 0, 0
    for i in range(matching_matrix.shape[0]):
        matching_matrix_sum += matching_matrix[:,i][targets_old[index_old]==targets_old[index_old][i]].sum()
        matching_matrix_old_sum += matching_matrix_old[:,i][targets_old[index_old]==targets_old[index_old][i]].sum()
    print('matching_matrix', matching_matrix_sum)
    print('matching_matrix_old',matching_matrix_old_sum)
    '''
    return {'matching_loss_seed': matching_loss_seed}


def matchingmatrix(P, alpha, t=2):
    matrix = ((1-alpha)**t) * torch.matrix_power(P, t)  # 
    return matrix

def matchingmatrix_inverse(P, alpha, t=2):
    I = torch.eye(P.shape[0]).to(P.device)
    matrix = ((1-alpha)**t) * torch.inverse(I-P)  # 
    return matrix


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
