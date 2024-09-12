import numpy as np
from numpy import array
import os,sys,pickle,torch,ot
from functools import reduce
from fugw.mappings import FUGW
from scipy.spatial import distance_matrix
from pathlib import Path
from torch import tensor
from torch import ones,zeros
cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cfd+'/..')
sys.path.append(cfd+'/../..')



from myWD.FUGW import my_FUGW
from myWD.AWD import my_partialRWD


def my_miccai_RWD_BC(features_list,locations_list,weights_list,features_BC,locations_BC,weights_BC,zeta_a=0,zeta_b=0,lam=0.5,iter_BC = 5,iter_RWD=5,clean_rot_locations_list=None):
    for i in range(iter_BC):
        new_locations_list = []
        all_features = features_BC * 0
        all_features_scaler = weights_BC * 0
        BC_loss = 0
        Pa_list = []
        for fea_a,locs_a,weights_a in zip(features_list,locations_list,weights_list):
            tmp_fea_a = fea_a   #;   tmp_fea_a = tmp_fea_a.reshape((tmp_fea_a.shape[0],1))
            tmp_features_BC = features_BC #; tmp_features_BC = tmp_features_BC.reshape((tmp_features_BC.shape[0],1))
            tmp_flow_matrix,geometry_loss,feature_loss,tmp_loss,my_corr,diff_a,diff_b,tmp_locations = my_partialRWD(tmp_fea_a,locs_a,weights_a,tmp_features_BC,locations_BC,weights_BC,zeta_a,zeta_b,lam,iter_num=iter_RWD)
            Pa_list.append(tmp_flow_matrix.sum(dim=1))
            new_locations_list.append(tmp_locations)
            all_features = all_features + torch.matmul(tmp_flow_matrix.T.to(torch.float64),fea_a)
            all_features_scaler = all_features_scaler + tmp_flow_matrix.sum(dim=0)
            BC_loss = BC_loss + tmp_loss
        features_BC = tensor(np.array([fea/fea_scalar for fea,fea_scalar  in zip(all_features,all_features_scaler)])).nan_to_num()  
        locations_list = new_locations_list  
        print("BC_loss = ",BC_loss)
    diff_a_list = [torch.abs(wei_t - Pa_t).sum() for wei_t,Pa_t in zip(weights_list,Pa_list)]
    diff_mean = np.array(diff_a_list).mean()
    print("==========  diff_mean = ",diff_mean)
    return features_BC,locations_BC,weights_BC,diff_mean,BC_loss
        
    
    
    
def test_my_miccai_RWD_BC():    
    zeta_a = 0; zeta_b = 0; miccai_theta = 0.5
    fea_index=360;n_validation_contrasts=40
    subject_num = 3
    downsample_file_path = cfd + '/../my_miccai_getdata/my_downsample_data/my_IBP_scaleFactor10_subject13_1573'
    with open(downsample_file_path, 'rb') as file:
        downsample_data = pickle.load(file)
    print('downsample_data = ',len(downsample_data))
    downsample_data = downsample_data[:subject_num]
    features_list,locations_list,weights_list = downsample_data[0],downsample_data[1],downsample_data[2]
    features_BC,locations_BC,weights_BC = features_list[0],locations_list[0],weights_list[0]
    fea_index = 0
    # features_list = array([fea[:,fea_index] for fea in features_list])
    # features_BC = features_BC[:,fea_index]
    #features_BC,locations_BC, = features_BC[:10],locations_BC[:10]
    #weights_BC = np.ones(len(locations_BC)) / len(locations_BC)
    #locations_BC = locations_BC - weights_BC.dot(locations_BC)
    my_miccai_RWD_BC(features_list,locations_list,weights_list,features_BC,locations_BC,weights_BC,fea_index,zeta_a,zeta_b,miccai_theta,num_iter = 5)    
    
    
    
    
#test_my_miccai_RWD_BC()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



def my_miccai_FUGW_BC(features_list,locations_list,weights_list,features_BC,locations_BC,weights_BC,lam=0.5,rho=1,eps=0.0001,my_solver='sinkhorn',num_iter = 5):

    # features_BC = zeros(array(features_list[0]).shape) 
    geometry_BC = ot.dist(locations_BC,locations_BC)  
    # locations_BC = locations_list[0]
    # weights_BC = weights_list[0]
    
    for i in range(num_iter):
        flow_matrix_list = []
        # fea_a_normalized_list = []
        features_BC = features_BC.T
        Pa_list = []
        BC_loss_list = []
        for fea_a,locs_a,weights_a in zip(features_list,locations_list,weights_list):
            #new_locations_list.append(tmp_locations)
            fea_b = features_BC #/ np.linalg.norm(features_BC)
            geometry_a = distance_matrix(locs_a,locs_a)
            mapping = FUGW(alpha=lam, rho=rho, eps=eps)
            if my_solver == 'sinkhorn':
                _ = mapping.fit(
                    fea_a.T,
                    fea_b,
                    source_geometry=geometry_a,
                    target_geometry=geometry_BC,
                    solver="sinkhorn",
                    device = 'auto',
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
                    target_geometry=geometry_BC,
                    solver="mm",
                    device = 'auto',
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
                    target_geometry=geometry_BC,
                    solver="ibpp",
                    device = 'auto',
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
            flow_matrix_list.append(init_plan_normalized)
            # fea_a_normalized_list.append(fea_a)
            print("====")
            Pa_list.append(tensor(init_plan_normalized).sum(dim=1))
            #-------
            feature_cost = ot.dist(fea_a,tensor(fea_b).T)
            geometry_cost = (ot.dist(locs_a,locations_BC))
            costmatrix = geometry_cost + lam*feature_cost
            tmp_BC_loss = (costmatrix*init_plan_normalized).sum()
            BC_loss_list.append(tmp_BC_loss)    
        print("BC_loss = ",sum(BC_loss_list))
        all_features_BC_list = [np.diag(1/np.sum(flow,axis=0)).dot(flow.T).dot(fea_a) for flow,fea_a in zip(flow_matrix_list,features_list)]
        #features_BC = zeros(array(features_list[0]).shape) 
        features_BC = reduce(lambda x,y:x+y,all_features_BC_list); features_BC = features_BC / len(features_list)
        tmp_geometry_BC_list = [flow.T.dot(distance_matrix(loc_a,loc_a)).dot(flow) / (np.sum(flow,axis=0).dot(np.sum(flow,axis=0).T)) for flow,fea_a,loc_a in zip(flow_matrix_list,features_list,locations_list)]
        tmp_geometry_BC_list = tensor(tmp_geometry_BC_list)
        geometry_BC = reduce(lambda x,y:x+y,tmp_geometry_BC_list); geometry_BC = geometry_BC / len(features_list)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    diff_a_list = [torch.abs(wei_t - Pa_t).sum() for wei_t,Pa_t in zip(weights_list,Pa_list)]
    diff_mean = np.array(diff_a_list).mean()
    BC_loss = sum(BC_loss_list)
    print("==========  diff_mean = ",diff_mean)
    print("BC_loss = ",BC_loss)
    return tensor(features_BC),locations_BC,weights_BC,diff_mean,BC_loss

    
    
    
    
    
    
    
def test_my_miccai_FUGW_BC():    
    
    fea_index = 360; n_validation_contrasts = 40
    rho =100000 ; lam = 0.5; eps = 0.001
    subject_num = 3
    downsample_file_path = cfd + '/../my_miccai_getdata/my_downsample_data/my_IBP_scaleFactor10_subject13_1573'
    #downsample_file_path = cfd + '/../my_miccai_getdata/my_downsample_data/my_IBP_scaleFactor50_subject13_27'

    with open(downsample_file_path, 'rb') as file:
        downsample_data = pickle.load(file)
    print('downsample_data = ',len(downsample_data))
    downsample_data = downsample_data[:subject_num]
    features_list,locations_list,weights_list = downsample_data[0],downsample_data[1],downsample_data[2]
        
    fea_index = 0
    features_list = array([fea[:,fea_index] for fea in features_list])

    features_BC,locations_BC,weights_BC = my_miccai_FUGW_BC(features_list,locations_list,weights_list,fea_index,lam,rho,eps,my_solver='ibpp',num_iter = 5)

    print('==========')
    
#test_my_miccai_FUGW_BC()   
    
    
    
    
    
    