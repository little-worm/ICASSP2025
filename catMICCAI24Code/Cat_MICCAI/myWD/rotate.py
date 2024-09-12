
import numpy as np
from scipy.spatial.transform import Rotation as R

# 定义绕每个轴的旋转角度（以度为单位）



def my_3D_rotate(angles):
    # 将角度转换为弧度
    angles_rad = np.radians(angles)
    print("angles_rad = ",angles_rad)
    # 生成旋转对象
    rotation = R.from_euler('xyz', angles_rad)
    # 获取旋转矩阵
    rotation_matrix = rotation.as_matrix()
    # print("旋转矩阵：\n", rotation_matrix)
    return rotation_matrix

angles = [30, 45, 60]  # x轴, y轴, z轴的旋转角度

# my_3D_rotate(angles)