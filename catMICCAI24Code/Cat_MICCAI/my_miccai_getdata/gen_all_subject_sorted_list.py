import numpy as np
from gen_my_ibc_data import my_traverse_directory




def my_find_fileList_accordingTo_strList(str_list,file_list):
    org_data_list = []
    for str in str_list:
        tmp_data = None
        for file in file_list:
            if str in file:
                tmp_data = file
        assert not(tmp_data==None),"my file not in dataset"
        org_data_list.append(tmp_data)
    return org_data_list
    



def my_save_str_matrix(file_path,matrix):
    with open(file_path, 'w') as file:
        for row in matrix:
            line = '\t'.join(row) + '\n'  
            file.write(line)
    print("已成功保存数据到文件")    




def my_all_subsjects_sorted_list(save_path='./',file_name='all_subject_sorted_list.txt'):
    my_subject_list=['01', '02', '04', '05', '06', '07', '08', '09', '11', '12', '13', '14', '15']
    _,str_list = my_traverse_directory('my_ibc_data/sub-01')
    str_list = [ref[13:] for ref in str_list]
    #print(len(str_list),str_list[0])
    all_subsjects_sorted_list = []
    org_data_path = 'my_ibc_data/sub-'
    for sub in my_subject_list:
        tmp_path = org_data_path + sub
        file_list,_ = my_traverse_directory(tmp_path)
        #print('tmp_list = ',len(file_list) )
        res = my_find_fileList_accordingTo_strList(str_list,file_list)
        #print((res[20]))
        all_subsjects_sorted_list.append(res)
        file_path = save_path + file_name
        my_save_str_matrix(file_path, all_subsjects_sorted_list)
    return all_subsjects_sorted_list

def test_all_subsjects_sorted_list():
    all_subsjects_sorted_list = my_all_subsjects_sorted_list()



my_all_subsjects_sorted_list()
