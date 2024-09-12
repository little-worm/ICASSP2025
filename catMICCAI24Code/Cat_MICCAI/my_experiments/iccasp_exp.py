import sys,os,socket,argparse,random,wandb,gym,time
from pathlib import Path
import numpy as np
from numpy import array
import nibabel as nib
import torch,ot
from torch import tensor,ones
from nilearn import image, plotting
from nibabel import Nifti1Image
from matplotlib.colors import LinearSegmentedColormap


cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cfd+'/..')
sys.path.append(cfd)
#print(sys.path)
from my_miccai_getdata.read_IBC_data import read_all_subjects,read_all_subjects_Pool
# from myWD.add_noise import add_artifact
from myWD.rotate import my_3D_rotate
from myWD.FUGW import my_FUGW
from myWD.AWD import my_partialRWD
from myWD.miccaiBC import my_miccai_RWD_BC,my_miccai_FUGW_BC
def parse(args):
    parser = argparse.ArgumentParser(
        description='iccasp-24', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team_name",default="worm_tean", type=str)
    parser.add_argument("--project_name", default="iccasp", type=str)
    parser.add_argument("--scenario_name", default="barycenter", type=str, choices=["distance","barycenter"])
    parser.add_argument("--experiment_name", default="FUGW", type=str,choices=["FUGW","RWD","WD"])
    parser.add_argument("--seed",type=int,default=0)
    
    parser.add_argument("--my_subject_num",type=int,default=3)
   
    parser.add_argument("--my_SCALE_FACTOR",type=int,default=3)
    parser.add_argument("--my_zeta_a",type=float,default=0.1)
    parser.add_argument("--my_zeta_b",type=float,default=0.1)
    parser.add_argument("--my_lam",type=float,default=1)
    parser.add_argument("--plot_fea_index",type=int,default=100)
    parser.add_argument("--plot_threshold",type=str,default="0%")

    # parser.add_argument("--my_rho",type=float,default=1) # for FUGW

    all_args = parser.parse_known_args(args)[0]
    return all_args

def my_experiment(args):
    all_args = parse(args)

    random.seed(all_args.seed)

    run_dir = Path("results") / all_args.project_name / all_args.scenario_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))




    SCALE_FACTOR = all_args.my_SCALE_FACTOR; my_device = 'cpu'; zeta_a = all_args.my_zeta_a; zeta_b = all_args.my_zeta_b; my_lam = all_args.my_lam
    my_rho = None; noise_fea_value = 30; locs_scalar=10
    fmri_img_dataset = read_all_subjects(SCALE_FACTOR,all_args.my_subject_num)
    template_locations = None
    noisy_fea_list = []; noisy_locs_list = []; noisy_wei_list = []
    clean_fea_list = []; clean_locs_list = []
    for fmri_img,i in zip(fmri_img_dataset,range(fmri_img_dataset.shape[0])):
        fmri_img = fmri_img.nan_to_num()
        fmri_img = fmri_img.permute(1,2,3,0)
        coordinates = torch.tensor([(i,j,k) for i in range(fmri_img.shape[0]) for j in range(fmri_img.shape[1]) for k in range(fmri_img.shape[2])],dtype=torch.float64).to(my_device)
        org_fea = fmri_img.reshape(-1,fmri_img.shape[-1])
        tmp = torch.sum(org_fea,axis=1).to(my_device)
        clean_index = torch.tensor([i for i in range(tmp.shape[0]) if tmp[i] > 0 or np.isnan(tmp[i])])
        ### add noise---start
        noise_num = round(clean_index.shape[0]/(1-zeta_a)*zeta_a)
        valumn_index = torch.tensor([i for i in range(tmp.shape[0]) if not(tmp[i] > 0 or np.isnan(tmp[i]))])
        # valumn_index = torch.tensor([i*fmri_img.shape[1]*fmri_img.shape[2] + j*fmri_img.shape[2] + k for i in range(fmri_img.shape[0]) for j in range(fmri_img.shape[1]) for k in range(fmri_img.shape[2]) if i*j*k==0 or i==fmri_img.shape[0]-1 or j==fmri_img.shape[1]-1 or k==fmri_img.shape[2]-1 ]).to(my_device)


        random_index = np.random.choice(valumn_index)
        reference_point = coordinates[random_index]
        # reference_point = coordinates[0]
        # 3. 计算所有点到参考点的欧几里得距离
        distances = np.linalg.norm(coordinates - reference_point, axis=1)
        tmp_distances = np.zeros_like(distances); tmp_distances[clean_index] = 10000000
        distances = tmp_distances + distances
        print(distances)  
        closest_indices = np.argsort(distances)[:noise_num]  # 获取距离最近的100个点的索引
        org_fea[closest_indices] = noise_fea_value
        ### add noise---end
        
        tmp = torch.sum(org_fea,axis=1).to(my_device)
        noisy_index = torch.tensor([i for i in range(tmp.shape[0]) if tmp[i] > 0 or np.isnan(tmp[i])])
        clean_fea = org_fea[clean_index].clone(); clean_locs = coordinates[clean_index].clone()
        if i==0:
            template_locations = coordinates[clean_index].clone()
        clean_fea = clean_fea / 50; clean_locs = clean_locs / clean_locs.max() * locs_scalar
        clean_fea = clean_fea.nan_to_num()
        noisy_fea = org_fea[noisy_index]; noisy_locs = coordinates[noisy_index]
        noisy_fea = noisy_fea / 50; noisy_locs = noisy_locs / noisy_locs.max() * locs_scalar
        noisy_fea = torch.nan_to_num(noisy_fea)
        # locs = locs @ (torch.tensor(rotation_matrix).to(my_device))
        noisy_wei = torch.ones(noisy_fea.shape[0]) / noisy_fea.shape[0]
        noisy_fea_list.append(noisy_fea); noisy_locs_list.append(noisy_locs); noisy_wei_list.append(noisy_wei)
        clean_fea_list.append(clean_fea); clean_locs_list.append(clean_locs)
        print("")





    noise_fea_list = []; noise_locs_list = []; noise_wei_list = []

    for tmp_fea,tmp_locs,tmp_wei in zip(noisy_fea_list,noisy_locs_list,noisy_wei_list):
        # tmp_fea = add_artifact(tmp_locs,tmp_fea,zeta_a)
        angles = np.random.rand(3)*360
        rotation_matrix = my_3D_rotate(angles)
        print(rotation_matrix)
        tmp_locs = torch.matmul(tmp_locs,tensor(rotation_matrix))
        # tmp_fea = tmp_fea / 10        
        # tmp_locs = tmp_locs / tmp_locs.max() / 1.7
        noise_fea_list.append(tmp_fea); noise_locs_list.append(tmp_locs); noise_wei_list.append(tmp_wei)
            
            
            
            
    pairs_ij = [(i,j) for i in range(len(noise_fea_list)) for j in range(len(noise_fea_list)) if i>j]           
    fea_loss_matrix = np.zeros((len(noise_fea_list),len(noise_fea_list))); geo_loss_matrix = np.copy(fea_loss_matrix); loss_matrix = np.copy(geo_loss_matrix)

    # for tmp_exp_name in ["FUGW","RWD","RWD_no_rigid_iterRWD_no_rigid_iter"]:
    if all_args.scenario_name == "distance":
        for tmp_paras in [["FUGW",0.015625],["FUGW",0.03125],["FUGW",0.0625],["FUGW",0.125],["FUGW",0.25],["FUGW",0.5],["FUGW",1],["RWD"],["WD"]]:
            all_args.experiment_name = tmp_paras[0]
            wandb.init(config=all_args,
                project=all_args.project_name,
                entity=all_args.team_name,
                notes=socket.gethostname(),
                name=all_args.experiment_name+"_"+str(all_args.seed),
                group=all_args.scenario_name,
                dir=str(run_dir),
                job_type="training",
                reinit=True)
            start_time = time.time() 
            corr_list = []; geometry_loss_list = []; feature_loss_list = []; loss_list = []; diff_a_list = []; diff_b_list = []
            for ij in pairs_ij:
                print(ij)
                i,j = ij[0],ij[1]
                fea_a,locs_a,wei_a,fea_b,locs_b,wei_b = noise_fea_list[i],noise_locs_list[i],noise_wei_list[i],noise_fea_list[j],noise_locs_list[j],noise_wei_list[j]
                if all_args.experiment_name == "FUGW":
                    my_rho = tmp_paras[1]
                    init_plan_normalized,geometry_loss,feature_loss,loss,my_corr,diff_a,diff_b = my_FUGW(fea_a,locs_a,wei_a,fea_b,locs_b,wei_b,my_solver='sinkhorn',lam=my_lam,rho=my_rho,eps=0.001,my_device='auto')
                if all_args.experiment_name == "RWD":
                    my_rho = None
                    angles = np.random.rand(3)*360
                    rotation_matrix = my_3D_rotate(angles)
                    locs_b = torch.matmul(locs_b,tensor(rotation_matrix))
                    P,geometry_loss,feature_loss,loss,my_corr,diff_a,diff_b,_ = my_partialRWD(fea_a,locs_a,wei_a,fea_b,locs_b,wei_b,zeta_a,zeta_b,lam=my_lam,iter_num=5)
                if all_args.experiment_name == "WD":
                    my_rho = None
                    angles = np.random.rand(3)*360
                    rotation_matrix = my_3D_rotate(angles)
                    locs_b = torch.matmul(locs_b,tensor(rotation_matrix))
                    P,geometry_loss,feature_loss,loss,my_corr,diff_a,diff_b,_ = my_partialRWD(fea_a,locs_a,wei_a,fea_b,locs_b,wei_b,zeta_a,zeta_b,lam=my_lam,iter_num=1)

                corr_list.append(my_corr); geometry_loss_list.append(geometry_loss); feature_loss_list.append(feature_loss)
                loss_list.append(loss); diff_a_list.append(diff_a); diff_b_list.append(diff_b)
                fea_loss_matrix[i,j] = feature_loss; geo_loss_matrix[i,j] = geometry_loss; loss_matrix[i,j] = loss
                # 计算矩阵的均值
            fea_mean = np.mean(fea_loss_matrix); fea_std = np.std(fea_loss_matrix); fea_z_score_matrix = (fea_loss_matrix - fea_mean) / fea_std
            geo_mean = np.mean(geo_loss_matrix); geo_std = np.std(geo_loss_matrix); geo_z_score_matrix = (geo_loss_matrix - geo_mean) / geo_std
            mean = np.mean(loss_matrix); std = np.std(loss_matrix); z_score_matrix = (loss_matrix - mean) / std
            # print(z_score_matrix,geo_z_score_matrix,fea_z_score_matrix)
            end_time = time.time() 
            execution_time = end_time - start_time
            wandb.log({'corr':array(corr_list).mean(),'corr_std':array(corr_list).std(),'geo_loss':array(geometry_loss_list).mean(), 'geo_loss_std':array(geometry_loss_list).std(), 
                        "fea_loss":array(feature_loss_list).mean(), "fea_loss_std":array(feature_loss_list).std(),
                        "loss":array(loss_list).mean(), "loss_std":array(loss_list).std(),
                        "diff_a":array(diff_a_list).mean(), "diff_a_std":array(diff_a_list).std(),
                        "diff_b":array(diff_b_list).mean(),"diff_b_std":array(diff_b_list).std(), "execution_time":execution_time, "noise_fea_value":noise_fea_value,"rho":my_rho,"locs_scalar":locs_scalar
                        })
            wandb.finish()



    
    
    
    # if all_args.scenario_name == "barycenter":
    #     # html_view = None
    #     html_content = """
    #                     <!DOCTYPE html>
    #                     <html lang="en">
    #                     <head>
    #                         <meta charset="UTF-8">
    #                         <meta name="viewport" content="width=device-width, initial-scale=1.0">
    #                         <title>Nifti Images on Surface</title>
    #                     </head>
    #                     <body>
    #                     """

    #     fea_BC_0 = clean_fea_list[0] #; fea_BC_0 = torch.zeros_like(fea_BC_0)
    #     locs_BC_0 = clean_locs_list[0]
    #     weis_BC_0 = ones(fea_BC_0.shape[0]) / fea_BC_0.shape[0]
    #     # for tmp_paras in [["RWD"],["WD"]]:
    #     para_list = [["WD"],["RWD"],["FUGW",2],["FUGW",4],["FUGW",6],["FUGW",8],["FUGW",10],["FUGW",12],["FUGW",14],["FUGW",16],["FUGW",18],["FUGW",20],["FUGW",22],["FUGW",24]]
    #     # locs_scalar = 10
    #     para_list = [["WD"],["RWD"],["FUGW",4],["FUGW",8],["FUGW",12],["FUGW",16],["FUGW",20],["FUGW",24],["FUGW",28],["FUGW",32],["FUGW",36],["FUGW",40]]
    #     # para_list = [["WD"],["RWD"],["FUGW",24]]

    #     for tmp_paras in para_list:
    #         all_args.experiment_name = tmp_paras[0]
    #         wandb.init(config=all_args,
    #             project=all_args.project_name,
    #             entity=all_args.team_name,
    #             notes=socket.gethostname(),
    #             name=all_args.experiment_name+"_"+str(all_args.seed),
    #             group=all_args.scenario_name,
    #             dir=str(run_dir),
    #             job_type="training",
    #             reinit=True)
    #         start_time = time.time() 
    #         if all_args.experiment_name == "FUGW":
    #             my_rho = tmp_paras[1]
    #             features_BC,locations_BC,weights_BC,diff_mean,BC_loss = my_miccai_FUGW_BC(noise_fea_list,noise_locs_list,noise_wei_list,fea_BC_0,locs_BC_0,weis_BC_0,my_lam,rho=my_rho)
    #         if all_args.experiment_name == "RWD":
    #             print("==================RWD")
    #             # rotation_matrix = my_3D_rotate([100,200,300])
    #             # locs_BC_0 = torch.matmul(locs_BC_0,tensor(rotation_matrix))
    #             my_rho = None
    #             features_BC,locations_BC,weights_BC,diff_mean,BC_loss = my_miccai_RWD_BC(noise_fea_list,noise_locs_list,noise_wei_list,fea_BC_0,locs_BC_0,weis_BC_0,zeta_a,zeta_b,my_lam,iter_RWD=5)
    #         if all_args.experiment_name == "WD":
    #             print("================== WD")
    #             angles = np.random.rand(3)*360
    #             rotation_matrix = my_3D_rotate(angles)
    #             locs_BC_0 = torch.matmul(locs_BC_0,tensor(rotation_matrix))
    #             my_rho = None
    #             features_BC,locations_BC,weights_BC,diff_mean,BC_loss = my_miccai_RWD_BC(noise_fea_list,noise_locs_list,noise_wei_list,fea_BC_0,locs_BC_0,weis_BC_0,zeta_a,zeta_b,my_lam,iter_RWD=1)
    #         loss_list = []
    #         # locations_BC = (locations_BC - torch.mean(locations_BC,dim=0)).float()
    #         for fea_a,locs_a in zip(clean_fea_list,clean_locs_list):
    #             print(fea_a.shape,locs_a.shape,features_BC.shape,locations_BC.shape)
    #             # locs_a = (locs_a - torch.mean(locs_a,dim=0)).float()
    #             geometry_cost = ot.dist(locs_a,locations_BC)
    #             feature_cost = ot.dist(fea_a,features_BC)
    #             tmp_costmatrix = geometry_cost + my_lam * feature_cost
    #             t_wei_a = torch.ones(tmp_costmatrix.shape[0]) / tmp_costmatrix.shape[0]
    #             t_wei_b = torch.ones(tmp_costmatrix.shape[1]) / tmp_costmatrix.shape[1]
    #             tmp = ot.emd2(t_wei_a,t_wei_b,tmp_costmatrix)
    #             tmp_p = ot.emd(t_wei_a,t_wei_b,tmp_costmatrix)
    #             print("+++++++++++++++++++++",(tmp_p*geometry_cost).sum() , (my_lam *tmp_p*feature_cost).sum())
    #             loss_list.append(tmp)
    #         end_time = time.time() 
    #         execution_time = end_time - start_time
    
    
    
    
    
            
                        
    #         tmp_fmri_img = image.load_img('my_ibc_data/sub-01/sub-01_ses-00_task-ArchiSocial_dir-ap_space-MNI152NLin2009cAsym_desc-preproc_ZMap-false_belief_audio.nii.gz')
    #         target_affine = tmp_fmri_img.affine.copy() * SCALE_FACTOR
    #         tmp_fmri_data = tmp_fmri_img.get_fdata()
    #         tmp_fmri_data = tmp_fmri_data[::SCALE_FACTOR,::SCALE_FACTOR,::SCALE_FACTOR]
    #         BC_z_scores_voxel = np.zeros(tmp_fmri_data.shape)
    #         x_coor = template_locations[:,0].to(torch.int); y_coor = template_locations[:,1].to(torch.int); z_coor = template_locations[:,2].to(torch.int)
    #         z_scores = (features_BC - features_BC.mean()) / features_BC.std()
    #         BC_z_scores_voxel[x_coor,y_coor,z_coor] = z_scores[:,all_args.plot_fea_index]
    #         nii_img_RWD = Nifti1Image(BC_z_scores_voxel, header=None,affine=target_affine)
    #         # downsampled_img = image.resample_img(BC_z_scores_voxel, target_affine=target_affine)
            
    #         tmp_title = all_args.experiment_name + ", \u03B6" + "=" + str(zeta_a) + ", \u03C1" + "=" + str(my_rho) 
    #         view = plotting.view_img_on_surf(nii_img_RWD,threshold=all_args.plot_threshold,title=tmp_title,vmax=10,vmin=-10)
    #         iframe_html = view.get_iframe()
    #         html_content += f"<h2>Image {i + 1}</h2>\n"
    #         html_content += iframe_html
    #         # if html_view is None:
    #         #     html_view = view
    #         # else:
    #         #     html_view = html_view + view
    #         # 显示图像
    #         view.open_in_browser()
    #         # 保存图像为 HTML
    #         print("loss ======================================== ",np.mean(loss_list))
    #         wandb.log({'loss_mean':np.mean(loss_list),'loss_std':np.std(loss_list), "execution_time":execution_time, "rho":my_rho, "diff_mean":diff_mean, "noise_fea_value":noise_fea_value,"BC_loss":BC_loss,"locs_scalar":locs_scalar })
    #         wandb.finish()
    #     html_content += """
    #                     </body>
    #                     </html>
    #                     """
    #     output_file = "res_html/brain_surface" + str(zeta_a) + "_" + str(locs_scalar) + ".html"
    #     with open(output_file, 'w') as f:
    #         f.write(html_content)

    #     print("已成功保存到",output_file)





   
    if all_args.scenario_name == "barycenter":
        html_content = """
                        <!DOCTYPE html>
                        <html lang="en">
                        <head>
                            <meta charset="UTF-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <title>Nifti Images on Surface</title>
                        </head>
                        <body>
                        """

        fea_BC_0 = clean_fea_list[0] #; fea_BC_0 = torch.zeros_like(fea_BC_0)
        locs_BC_0 = clean_locs_list[0]
        weis_BC_0 = ones(fea_BC_0.shape[0]) / fea_BC_0.shape[0]
        # for tmp_paras in [["RWD"],["WD"]]:
        para_list = [["RWD"],["WD"],["FUGW",2],["FUGW",4],["FUGW",6],["FUGW",8],["FUGW",10],["FUGW",12],["FUGW",14],["FUGW",16],["FUGW",18],["FUGW",20],["FUGW",22],["FUGW",24]]
        para_list = [["RWD"],["WD"],["FUGW",4],["FUGW",8],["FUGW",12],["FUGW",16],["FUGW",20],["FUGW",24],["FUGW",28],["FUGW",32],["FUGW",36],["FUGW",40]]
        # para_list = [["RWD"],["WD"],["FUGW",1],["FUGW",2],["FUGW",4],["FUGW",8],["FUGW",12],["FUGW",16],["FUGW",20],["FUGW",24],["FUGW",28],["FUGW",32]]
        # para_list = [["RWD"],["WD"],["FUGW",1],["FUGW",2],["FUGW",4],["FUGW",6],["FUGW",8],["FUGW",10]]
        para_list = [["RWD"],["WD"],["FUGW",12],["FUGW",16],["FUGW",20],["FUGW",24],["FUGW",28],["FUGW",32]]

        for tmp_paras in para_list:
            all_args.experiment_name = tmp_paras[0]
            wandb.init(config=all_args,
                project=all_args.project_name,
                entity=all_args.team_name,
                notes=socket.gethostname(),
                name=all_args.experiment_name+"_"+str(all_args.seed),
                group=all_args.scenario_name,
                dir=str(run_dir),
                job_type="training",
                reinit=True)
            start_time = time.time() 
            if all_args.experiment_name == "FUGW":
                my_rho = tmp_paras[1]
                features_BC,locations_BC,weights_BC,diff_mean,BC_loss = my_miccai_FUGW_BC(noise_fea_list,noise_locs_list,noise_wei_list,fea_BC_0,locs_BC_0,weis_BC_0,my_lam,rho=my_rho)
            if all_args.experiment_name == "RWD":
                # rotation_matrix = my_3D_rotate([100,200,300])
                # locs_BC_0 = torch.matmul(locs_BC_0,tensor(rotation_matrix))
                my_rho = None
                features_BC,locations_BC,weights_BC,diff_mean,BC_loss = my_miccai_RWD_BC(noise_fea_list,noise_locs_list,noise_wei_list,fea_BC_0,locs_BC_0,weis_BC_0,zeta_a,zeta_b,my_lam,iter_RWD=5)
            if all_args.experiment_name == "WD":
                angles = np.random.rand(3)*360
                rotation_matrix = my_3D_rotate(angles)
                locs_BC_0 = torch.matmul(locs_BC_0,tensor(rotation_matrix))
                my_rho = None
                features_BC,locations_BC,weights_BC,diff_mean,BC_loss = my_miccai_RWD_BC(noise_fea_list,noise_locs_list,noise_wei_list,fea_BC_0,locs_BC_0,weis_BC_0,zeta_a,zeta_b,my_lam,iter_RWD=1)
            loss_list = []
            # locations_BC = (locations_BC - torch.mean(locations_BC,dim=0)).float()
            for fea_a,locs_a in zip(clean_fea_list,clean_locs_list):
                print(fea_a.shape,locs_a.shape,features_BC.shape,locations_BC.shape)
                # locs_a = (locs_a - torch.mean(locs_a,dim=0)).float()
                geometry_cost = ot.dist(locs_a,locations_BC)
                feature_cost = ot.dist(fea_a,features_BC)
                tmp_costmatrix = geometry_cost + my_lam * feature_cost
                t_wei_a = torch.ones(tmp_costmatrix.shape[0]) / tmp_costmatrix.shape[0]
                t_wei_b = torch.ones(tmp_costmatrix.shape[1]) / tmp_costmatrix.shape[1]
                tmp = ot.emd2(t_wei_a,t_wei_b,tmp_costmatrix)
                tmp_p = ot.emd(t_wei_a,t_wei_b,tmp_costmatrix)
                print("+++++++++++++++++++++",(tmp_p*geometry_cost).sum() , (my_lam *tmp_p*feature_cost).sum())
                loss_list.append(tmp)
            end_time = time.time() 
            execution_time = end_time - start_time
            
      
      
                     
            tmp_fmri_img = image.load_img('my_ibc_data/sub-01/sub-01_ses-00_task-ArchiSocial_dir-ap_space-MNI152NLin2009cAsym_desc-preproc_ZMap-false_belief_audio.nii.gz')
            target_affine = tmp_fmri_img.affine.copy() * SCALE_FACTOR
            tmp_fmri_data = tmp_fmri_img.get_fdata()
            tmp_fmri_data = tmp_fmri_data[::SCALE_FACTOR,::SCALE_FACTOR,::SCALE_FACTOR]
            BC_z_scores_voxel = np.zeros(tmp_fmri_data.shape)
            x_coor = template_locations[:,0].to(torch.int); y_coor = template_locations[:,1].to(torch.int); z_coor = template_locations[:,2].to(torch.int)
            z_scores = (features_BC - features_BC.mean()) / features_BC.std()
            BC_z_scores_voxel[x_coor,y_coor,z_coor] = z_scores[:,all_args.plot_fea_index]
            nii_img_RWD = Nifti1Image(BC_z_scores_voxel, header=None,affine=target_affine)
            # downsampled_img = image.resample_img(BC_z_scores_voxel, target_affine=target_affine)
            
            tmp_title = all_args.experiment_name + ", \u03B6" + "=" + str(zeta_a) + ", \u03C1" + "=" + str(my_rho) + ", diff=" + str(format(diff_mean, '.2f'))
            # 自定义颜色映射
            colors = [(0, 'blue'), (0.5, 'white'), (1, 'red')]  # 在中间设置为白色
            cmap = LinearSegmentedColormap.from_list('custom_cmap', colors) 
            view = plotting.view_img_on_surf(nii_img_RWD,threshold=all_args.plot_threshold,title=tmp_title,title_fontsize=30,cmap=cmap,vmin=-3,vmax=3)
            iframe_html = view.get_iframe()
            html_content += f"<h2>Image {i + 1}</h2>\n"
            html_content += iframe_html
            # if html_view is None:
            #     html_view = view
            # else:
            #     html_view = html_view + view
            # 显示图像
            view.open_in_browser()
            # 保存图像为 HTML
            print("loss ======================================== ",np.mean(loss_list))
            wandb.log({'loss_mean':np.mean(loss_list),'loss_std':np.std(loss_list), "execution_time":execution_time, "rho":my_rho, "diff_mean":diff_mean, "noise_fea_value":noise_fea_value,"BC_loss":BC_loss,"locs_scalar":locs_scalar })
            wandb.finish()
        html_content += """
                        </body>
                        </html>
                        """
        output_file = "res_html/brain_surface" + str(zeta_a) + "_" + str(locs_scalar) + ".html"
        with open(output_file, 'w') as f:
            f.write(html_content)

        print("已成功保存到",output_file)

      
      
      
      
      
      
      
      
      
      
      
      
            
            # print("loss ======================================== ",np.mean(loss_list))
            # wandb.log({'loss_mean':np.mean(loss_list),'loss_std':np.std(loss_list), "execution_time":execution_time, "rho":my_rho, "diff_mean":diff_mean, "noise_fea_value":noise_fea_value,"BC_loss":BC_loss,"locs_scalar":locs_scalar })
            # wandb.finish()












if __name__ == '__main__':
    my_experiment(sys.argv[1:])
    
    
    
    
    
    
    
    
    
    
    
    
    
    