import os
import shutil


 
def my_traverse_directory(folder,contain_string=''):
    file_path_list = [] # 创建空数组用于保存文件路径
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if contain_string in file:
                file_path = os.path.join(root, file) # 获取完整的文件路径
                file_path_list.append(file_path) # 将文件路径添加到数组中   
                file_list.append(file)        
    return file_path_list,file_list
 

 
 
 
def my_mkdir(dir_name): 
    if not os.path.exists(dir_name):
        os.makedirs(dir_name) 






def my_substr_in_all_strs(my_str,my_strList):
    flag_list = []
    for t_strs in my_strList:        
        res = [ my_str in s for  s in t_strs ]
       # print('res = ',res)
        tmp_flag = ( sum(res) > 0 )
        #print(sum(res),len(my_strList),flag)
        flag_list.append(tmp_flag)
    #print('flag_list = ',flag_list)
    my_flag = (sum(flag_list) == len(my_strList))
    return my_flag


def test_my_substr_in_all_strs():
    my_str = 'aa'; my_strList = [['aaa','bbaab','ccc'],['aaa'],['a']]
    res = my_substr_in_all_strs(my_str,my_strList)
    print('res = ',res)

#test_my_substr_in_all_strs()







def my_mvFile(source_folder,target_folder,string1,string2="",string3="",string4=''):
    my_mkdir(target_folder)
    cou = 0
    file_path_list,_ = my_traverse_directory(source_folder)
    #print(source_folder)
    #print(file_path_list)
    for filename in file_path_list:
        if string1 in filename and string2 in filename and string3 in filename and string4 in filename:
            #print("===",filename)
            source_filepath = os.path.join(source_folder, filename)
            source_filepath = filename
            if not os.path.isdir(source_filepath):
                shutil.copy2(source_filepath, target_folder)
                cou = cou + 1
                #print("cou = ",cou)
                return


         



def test_my_mvFile():
    # 源文件夹路径
    source_folder = "my_ibc_data/tmpSub0/sub-01"
    # 目标文件夹路径
    target_folder = "ha"
    string1 = "ArchiSocial"
    string2 = "audio"
    string3 = ".nii.gz"
    my_mvFile(source_folder,target_folder,string1,string2)




if __name__ == '__main__':
    pass
    # test_my_mvFile()


"""

my_subject_list=['01', '02', '04', '05', '06', '07', '08', '09', '11', '12', '13', '14', '15']

my_folder = 'ibc_data/volume_maps/sub-'
for sub in my_subject_list:
    tmp_my_folder = my_folder + sub
    tmp = len(my_traverse_directory(tmp_my_folder)[0])
#    print(sub,tmp)
    
    
    
    
my_subject_list=['01', '02', '04', '05', '06', '07', '08', '09', '11', '12', '13', '14', '15']
my_data_path = '../ibc_data/volume_maps/sub-'
my_data_path_list = [my_data_path + pa for pa in my_subject_list]
#print("my_data_path_list = ",my_data_path_list)
keep  = my_data_path + '08' 
_, ref_list = my_traverse_directory(keep,'.nii.gz')
ref_list = [ li[13:] for li in ref_list ]
#print("ref_list = ",len(ref_list))
#print(keep_list[1])
all_data_list = []
for dpl in my_data_path_list:
    _,tmp = my_traverse_directory(dpl,'nii.gz')
    all_data_list.append(tmp)
    #print("===",len(tmp))
    
keep_list = []

for rl in ref_list:   
    flag = my_substr_in_all_strs(rl,all_data_list)
    if flag and not(rl in keep_list):
        keep_list.append(rl)        
            

print("keep_list = ",len(keep_list))





my_subject_list=['01', '02', '04', '05', '06', '07', '08', '09', '11', '12', '13', '14', '15']

#my_subject_list=['04']

my_data_path = '../ibc_data/volume_maps/sub-'
my_data_path_list = [my_data_path + pa for pa in my_subject_list]
my_sorted_data_path = 'my_ibc_data/sub-'
print("my_data_path_list = ",my_data_path_list)


for tmp_subjet in my_subject_list:
    print("----------------------------------")
    source_folder = my_data_path + tmp_subjet
    target_folder = my_sorted_data_path + tmp_subjet
    for string1 in keep_list:
        my_mvFile(source_folder,target_folder,string1)
    print(len(my_traverse_directory(target_folder)[1]))

"""               

                    
                    



                    