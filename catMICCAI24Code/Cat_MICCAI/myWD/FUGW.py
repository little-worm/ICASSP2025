import ot,sys,os,pickle
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from fugw.mappings import FUGW
from fugw.utils import load_mapping, save_mapping
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nilearn import datasets, image, plotting, surface
import torch
from torch import tensor
from scipy.linalg import norm
from scipy.spatial import distance_matrix
cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cfd+'/..')
from .pearson import pearson_corr,pearson_r

cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cfd+'/..')
sys.path.append(cfd)





def my_FUGW(fea_a,locs_a,wei_a,fea_b,locs_b,wei_b,my_solver='sinkhorn',lam=0.5,rho=1,eps=0.001,my_device='auto'):
    assert type(fea_a) == torch.Tensor and type(locs_a) == torch.Tensor and type(wei_a) == torch.Tensor
    assert type(fea_b) == torch.Tensor and type(locs_b) == torch.Tensor and type(wei_b) == torch.Tensor
    fea_a = fea_a.nan_to_num(); fea_b = fea_b.nan_to_num()
    '''
    fixe "location_a", transform "location_b
    '''
    if my_device == 'cpu':
        my_device = torch.device('cpu')
    fea_a = fea_a.T; fea_b = fea_b.T
    geometry_a = distance_matrix(locs_a,locs_a)
    geometry_b = distance_matrix(locs_b,locs_b)

    mapping = FUGW(alpha=lam, rho=rho, eps=eps)
    if my_solver == 'sinkhorn':
        _ = mapping.fit(
            fea_a,
            fea_b,
            source_geometry=geometry_a,
            target_geometry=geometry_b,
            solver="sinkhorn",
            device = my_device,
            solver_params={
                "nits_bcd": 3,
            },
            verbose=True,
        )
    elif my_solver == 'mm':
        _ = mapping.fit(
            fea_a,
            fea_b,
            source_geometry=geometry_a,
            target_geometry=geometry_b,
            solver="mm",
            device = my_device,
            solver_params={
                "nits_bcd": 5,
                "tol_bcd": 1e-10,
                "tol_uot": 1e-10,
            },
            verbose=True,
        )
    elif my_solver == 'ibpp':
        _ = mapping.fit(
            fea_a,
            fea_b,
            source_geometry=geometry_a,
            target_geometry=geometry_b,
            solver="ibpp",
            device = my_device,
            solver_params={
                "nits_bcd": 5,
                "tol_bcd": 1e-10,
                "tol_uot": 1e-10,
            },
            verbose=True,
        )
    else:
        assert False, "undefined solver for FUGW!!!"

    pi = mapping.pi.numpy()
    pi = pi / sum(sum(pi))
    #print('pi = ',sum(sum(pi)))
    init_plan_normalized = pi / pi.sum()
    #print('pi = ',sum(sum(pi)))

    del_fea = ot.dist(fea_a.T,fea_b.T)
    del_geo = ot.dist(locs_a,locs_b)

    feature_loss = lam*sum(sum(tensor(pi) * ot.dist(fea_a.T,fea_b.T)))
    geometry_loss = sum(sum(tensor(pi)*ot.dist(locs_a,locs_b)))
    #print("ot.dist(fea_a.T,fea_b.T) = ",ot.dist(fea_a.T,fea_b.T))
    diff_a = sum(np.abs(np.sum(init_plan_normalized,axis=1) - np.array(wei_a)))
    diff_b = sum(np.abs(np.sum(init_plan_normalized,axis=0) - np.array(wei_b)))
    print('diff_a,diff_b = ', diff_a,diff_b)
    loss = geometry_loss + feature_loss
    print("geometry_loss,feature_loss,loss,del_geo.max(),del_fea.max()*lam = ")
    print(geometry_loss.item(),feature_loss.item(),loss.item(),del_geo.max().item(),(del_fea.max()*lam).item())
    my_corr = pearson_corr(fea_a,fea_b, init_plan_normalized); my_corr = array(my_corr).mean()
    print("my_corr-FUGW = ",(my_corr))        

    return init_plan_normalized,geometry_loss,feature_loss,loss,my_corr,diff_a,diff_b








def test_surface():
    import os,sys
    cfd = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(cfd+'/..')
    print(sys.path)
    #from my_read_IBC_data import my_read_IBC_data

    my_solver = 'mm'
    #all_subject_sorted_list = my_load_str_matrix('/data/cat/catMICCAI24Code/Cat_MICCAI/my_miccai_getdata/all_subject_sorted_list.txt')[:2]
    #print('all_subject_sorted_list = ',len(all_subject_sorted_list))
    downsample_file_path = cfd + '/../my_miccai_getdata/my_downsample_data/my_IBP_scaleFactor10_subject13_1573'
    with open(downsample_file_path, 'rb') as file:
        downsample_data = pickle.load(file)
    features_list,locations_list,weights_list  = downsample_data[0],downsample_data[1],downsample_data[2]

    SCALE_FACTOR = 20
    noise_para_list = []
    noise_para_list = [50,60,50,60,40]
    #features_list,locations_list,weights_list  = my_read_IBC_data(all_subject_sorted_list,SCALE_FACTOR,noise_para_list)
    my_mode='hyper_para_search'
    #my_mode='training'
    fea_a,fea_b = features_list[0],features_list[1]
    locs_a,locs_b = locations_list[0],locations_list[1]
    wei_a,wei_b = weights_list[0],weights_list[1]
    res = my_FUGW(fea_a,locs_a,wei_a,fea_b,locs_b,wei_b,my_solver='sinkhorn',lam=0.5,rho=1,eps=0.001,my_device='auto')
    print('res = ',res)





# test_surface()




