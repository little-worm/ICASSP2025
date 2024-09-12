import numpy as np
import ot
from nilearn import datasets
import nibabel as nib
import torch
from torch import tensor,zeros,ones
from pearson import pearson_corr,pearson_r










def my_partialRWD(fea_a,locs_a,wei_a,fea_b,locs_b,wei_b,zeta_a=0,zeta_b=0,lam=1,iter_num=5,my_device='cpu'):
    # fea_a = tensor(fea_a); locs_a = tensor(locs_a); wei_a = tensor(wei_a)
    # fea_b = tensor(fea_b); locs_b = tensor(locs_b); wei_b = tensor(wei_b)
    assert type(fea_a) == torch.Tensor and type(locs_a) == torch.Tensor and type(wei_a) == torch.Tensor
    assert type(fea_b) == torch.Tensor and type(locs_b) == torch.Tensor and type(wei_b) == torch.Tensor
    fea_a = fea_a.nan_to_num(); fea_b = fea_b.nan_to_num()
    size_a,size_b = fea_a.shape[0],fea_b.shape[0]
    robust_wei_a = zeros(size_a+1)
    robust_wei_a[:-1] = wei_a / (1-zeta_a); robust_wei_a[-1] = zeta_b / (1-zeta_b)
    robust_wei_b = zeros(size_b+1)
    robust_wei_b[:-1] = wei_b / (1-zeta_b); robust_wei_b[-1] = zeta_a / (1-zeta_a)
    robust_cost_matrix = zeros((size_a+1,size_b+1))

    feature_cost = ot.dist(fea_a,fea_b)
    # mean = 0
    locs_a = (locs_a - torch.mean(locs_a,dim=0)).float()
    locs_b = (locs_b - torch.mean(locs_b,dim=0)).float()
    #fix locs_b, rotate locs_a
    for iter in range(iter_num):
        geometry_cost = (ot.dist(locs_a,locs_b))
        costmatrix = geometry_cost + lam*feature_cost
        robust_cost_matrix[:-1,:-1] = costmatrix
        robust_P = ot.emd(robust_wei_a,robust_wei_b,robust_cost_matrix)
        P = robust_P[:-1,:-1]
        matrixB = (locs_a.T) @ P
        matrixB = matrixB @ locs_b
        matrixU,matrixS,matrixVT = torch.linalg.svd(matrixB)
        diagList = list([1 for i in range(len(matrixB)-1)])
        diagList.append(torch.linalg.det(matrixU)*torch.linalg.det(matrixVT))
        diagList = torch.tensor(diagList).to(my_device)
        matrixR = matrixU @ ( torch.diag(  diagList  ))
        matrixR = matrixR @ (matrixVT)
        if iter_num > 1:
            # print("=====rotate====iter_num",iter_num)
            locs_a = locs_a @( matrixR )
            #--------
            Pa = P.sum(axis=1); Pb = P.sum(axis=0)
            locs_a = locs_a - torch.matmul(Pa,locs_a) / Pa.sum()
            locs_b = locs_b - torch.matmul(Pb,locs_b) / Pb.sum()
        # print("loss = ",(P*costmatrix).sum())
    geometry_loss = (P * geometry_cost).sum().item()
    feature_loss = lam*(P * feature_cost).sum().item()
    loss = (P*costmatrix).sum().item()
    print("geometry_loss,feature_loss,loss,geometry_cost.max(),feature_cost.max() = ")
    print(geometry_loss,feature_loss,loss,geometry_cost.max().item(),feature_cost.max().item()*lam)
    # print("nan = ",np.isnan(fea_a).sum(),np.isnan(fea_b).sum())
    my_corr = pearson_corr(fea_a.T,fea_b.T, P); my_corr = np.array(my_corr).mean()
    diff_a = sum(torch.abs(P.sum(dim=1) - wei_a))
    diff_b = sum(torch.abs(P.sum(dim=0) - wei_b))
    print('diff_a,diff_b = ', diff_a,diff_b)
    # print("my_corr-RWD  = ",my_corr) 
    return P,geometry_loss,feature_loss,loss,my_corr,diff_a,diff_b,locs_a









def test_my_robustAWD():  
    n = 4    
    dim = 2
    my_method = 'highs'; my_x0 = None
    my_mode = 'rigid'
    #my_mode = 'affine'
    locs_a = np.random.rand(n, dim)
    wei_a = ones(locs_a.shape[0]) / locs_a.shape[0]
    locs_b = np.random.rand(n, dim)
    wei_b = ones(locs_b.shape[0]) / locs_b.shape[0]
    #----------------------
    n1 = 1000
    n2 = 100
    d = 1
    fea_a =  torch.rand(n1, dim)
    fea_b =  torch.rand(n2, dim)
    locs_a = torch.rand(n1,d)*100
    locs_b = torch.rand(n2,d)*100
    wei_a = torch.ones(locs_a.shape[0]) / locs_a.shape[0]
    wei_b = torch.ones(locs_b.shape[0]) / locs_b.shape[0]
    locs_a = locs_a - torch.matmul(wei_a,locs_a)
    locs_b = locs_b - torch.matmul(wei_b,locs_b)
    print(len(locs_a),len(locs_b))
    zeta_a = 0.1
    zeta_b = 0
    miccai_theta = 0.5
    my_partialRWD(fea_a,locs_a,wei_a,fea_b,locs_b,wei_b,zeta_a=0,zeta_b=0,lam=1,iter_num=3,my_device='cpu')

# test_my_robustAWD()