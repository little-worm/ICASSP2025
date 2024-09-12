import numpy as np

# 1. 生成10000个3D点
locs = np.random.rand(100, 3)  # 随机生成10000个3D点 (x, y, z) 坐标
fea = locs 
noise_mass = 0.1


def add_artifact(locs,fea,noise_mass):
    noise_num = round(noise_mass*np.array(locs).shape[0])
    print("noise_num = ",noise_num)
    # 2. 随机选择一个点作为参考点
    random_index = np.random.randint(0,np.array(fea).shape[0])
    reference_point = locs[random_index]
    reference_point = np.array([0,0,0])
    print("random_index = ",random_index)
    # 3. 计算所有点到参考点的欧几里得距离
    distances = np.linalg.norm(locs - reference_point, axis=1)
    print(distances)
    # 4. 按照距离排序，取出最近的100个点
    closest_indices = np.argsort(distances)[:noise_num]  # 获取距离最近的100个点的索引
    fea[closest_indices] = 0.5  # 提取距离最近的100个点的坐标
    return fea
