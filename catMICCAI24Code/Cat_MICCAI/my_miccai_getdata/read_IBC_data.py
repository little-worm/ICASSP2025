import numpy as np
from numpy import ones
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import time,sys,os,pickle,random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nilearn import datasets, image
from scipy.spatial import distance_matrix
from fugw.mappings import FUGW
import multiprocessing,torch
import nibabel as nib
from torch import tensor
cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append('/root/autodl-tmp/cat')
# sys.path.append(os.path.abspath(cfd+'/..'))
# sys.path.append(os.path.abspath(cfd+'/../..'))
sys.path.append(os.path.abspath(cfd+'/../../..'))

# print(sys.path)





def my_load_str_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()   
    result_matrix = []
    for line in lines:
        row = line.strip().split('\t')  # 去除换行符并根据制表符切片为单词
        result_matrix.append(row)   
    print("The data has been read successfully")
    #print(result_matrix)
    return result_matrix



def read_single_subject(arg):
    Nii_img = []
    img_path_list,SCALE_FACTOR = arg[0],arg[1]
    # cou = 0
    for img_path in img_path_list:
        # cou = cou + 1
        # print("--------------------------",cou)
        img = nib.load(img_path)
        data = img.get_fdata()
        Nii_img.append(data)
    Nii_img = tensor(np.array(Nii_img))[:,::SCALE_FACTOR,::SCALE_FACTOR,::SCALE_FACTOR]
    print(Nii_img.shape)
    return Nii_img




def test_read_single_subject():
    all_subject_sorted_list = my_load_str_matrix(cfd+'/all_subject_sorted_list.txt')
    read_single_subject([all_subject_sorted_list[0],10])

# test_read_single_subject()



# def my_read_IBC_data(all_subject_sorted_list,SCALE_FACTOR = 20):
#     features_list = []
#     locations_list = []
#     weights_list = []
#     for imgs_paths in all_subject_sorted_list:
#         # print("imgs_paths = ",os.path.abspath(imgs_paths[0]))
#         imgs = image.load_img(imgs_paths[0])
#         org_maps = imgs.get_fdata()
#         tmp_head = imgs.header
#         tmp_affine = imgs.affine
#         print('tmp_head = ',tmp_head)
#         print('tmp_affine = ',tmp_affine)
#         img_shape = org_maps.shape
#         print('img_shape = ',img_shape)
#         scaled_maps = np.nan_to_num(org_maps[::SCALE_FACTOR, ::SCALE_FACTOR, ::SCALE_FACTOR])
#         segmentation_fine = np.logical_not(np.isnan(org_maps[:, :, :, 0]))
#         segmentation_coarse = segmentation_fine[::SCALE_FACTOR, ::SCALE_FACTOR, ::SCALE_FACTOR]
#         locations = np.array(np.nonzero(segmentation_coarse)).T

#         features = scaled_maps[locations[:, 0], locations[:, 1], locations[:, 2]]
    #     features.shape

    #     print("=====",len(locations))
    #     print('locations = ',len(locations))
    #     weights = ones(locations.shape[0]) / locations.shape[0]
    #     locations = locations - weights.dot(locations)
    #     features_list.append(features)
    #     locations_list.append(locations)
    #     weights_list.append(weights)
    # res = [features_list,locations_list,weights_list]
    # res_save_path = cfd + '/my_downsample_data/my_IBP_scaleFactor'+str(SCALE_FACTOR) + '_subject'+str(len(all_subject_sorted_list))+'_'+str(len(weights_list[0]))

    # res_save_path = res_save_path 
    # with open(res_save_path, 'wb') as file:
    #     pickle.dump(res, file)
    # return features_list,locations_list,weights_list   
    


   




def read_all_subjects_Pool(SCALE_FACTOR,subject_num=13):

    all_subject_sorted_list = my_load_str_matrix(cfd+'/all_subject_sorted_list.txt')[:subject_num]
    #all_subject_sorted_list = all_subject_sorted_list[:6]
    #print('all_subject_sorted_list = ',len(all_subject_sorted_list))
    # features_list,locations_list,weights_list  = my_read_IBC_data(all_subject_sorted_list,SCALE_FACTOR)
    # print('SCALE_FACTOR, len(features_list) = ',SCALE_FACTOR,len(features_list))
    # arg = all_subject_sorted_list[0]
    # nii_filepath_list = ["/root/autodl-tmp/cat/ibc_data/volume_maps/sub-11/ses-00/sub-11_ses-00_task-ArchiSpatial_dir-ap_space-MNI152NLin2009cAsym_desc-preproc_ZMap-grasp-orientation.nii.gz",
    #                      "/root/autodl-tmp/cat/ibc_data/volume_maps/sub-11/ses-00/sub-11_ses-00_task-ArchiSpatial_dir-ap_space-MNI152NLin2009cAsym_desc-preproc_ZMap-hand-side.nii.gz",
    #                      "/root/autodl-tmp/cat/ibc_data/volume_maps/sub-11/ses-00/sub-11_ses-00_task-ArchiSpatial_dir-ap_space-MNI152NLin2009cAsym_desc-preproc_ZMap-object_grasp.nii.gz"
    #                    ]
    # arg = nii_filepath_list
    # res = read_single_subject(arg)
    # print(res.shape)
    # arg_list = [arg,arg,arg,arg,arg,arg]
    arg_list = [[imgs_l, SCALE_FACTOR] for imgs_l in all_subject_sorted_list]
    with multiprocessing.Pool(16) as pool:
        res_list = pool.map(read_single_subject,arg_list)
    
    print(tensor(res_list).shape)
        







def read_all_subjects(SCALE_FACTOR,subject_num=13):
    fmri_img_list = []
    all_subject_sorted_list = my_load_str_matrix(cfd+'/all_subject_sorted_list.txt')[:subject_num]
    for imgs_file in all_subject_sorted_list:
        fmri_img = read_single_subject([imgs_file,SCALE_FACTOR])
        fmri_img_list.append(fmri_img)
    fmri_img_list = torch.tensor(np.array(fmri_img_list))
    return fmri_img_list
    


if __name__ == '__main__':
    random.seed(666)
    for SCALE_FACTOR in range(10,11):
        # read_all_subjects_Pool(SCALE_FACTOR)
        read_all_subjects(SCALE_FACTOR)