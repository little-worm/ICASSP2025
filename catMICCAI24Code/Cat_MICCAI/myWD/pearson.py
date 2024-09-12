import gdist
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from fugw.mappings import FUGW
from fugw.utils import load_mapping, save_mapping
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nilearn import datasets, image, plotting, surface
import torch
from scipy.linalg import norm





def pearson_corr(a, b, plan):
    """
    Compute the Pearson correlation between transformed
    source features and target features.
    """
    if torch.is_tensor(a):
        x = a.detach().cpu().numpy()
    elif isinstance(a, np.ndarray):
        x = a
    elif isinstance(a, list):
        x = np.array(a)
    else:
        raise ValueError("a must be a list, np.ndarray or torch.Tensor")

    if torch.is_tensor(b):
        y = b.detach().cpu().numpy()
    elif isinstance(b, np.ndarray):
        y = b
    elif isinstance(b, list):
        y = np.array(b)
    else:
        raise ValueError("b must be a list, np.ndarray or torch.Tensor")

    # Compute the transformed features
    x_transformed = (
        (plan.T @ x.T ).T
        # (plan.T @ x.T / plan.sum(axis=0).reshape(-1, 1)).T #.detach().cpu()
    )
    # print("nan = ",np.isnan(a).sum(),np.isnan(b).sum(),np.isnan(plan).sum(),np.isnan(x_transformed).sum())

    return pearson_r(x_transformed, y)




def pearson_r(a, b):
    # print("nan = ",np.isnan(a).sum(),np.isnan(b).sum())
    """Compute Pearson correlation between x and y.
    
    Compute Pearson correlation between 2d arrays x and y
    along the samples axis.
    Adapted from scipy.stats.pearsonr.

    Parameters
    ----------
    a: np.ndarray of size (n_samples, n_features)
    b: np.ndarray of size (n_samples, n_features)

    Returns
    -------
    r: np.ndarray of size (n_samples,)
    """
    if torch.is_tensor(a):
        x = a.detach().cpu().numpy()
    elif isinstance(a, np.ndarray):
        x = a
    elif isinstance(a, list):
        x = np.array(a)
    else:
        raise ValueError("a must be a list, np.ndarray or torch.Tensor")

    if torch.is_tensor(b):
        y = b.detach().cpu().numpy()
    elif isinstance(b, np.ndarray):
        y = b
    elif isinstance(b, list):
        y = np.array(b)
    else:
        raise ValueError("b must be a list, np.ndarray or torch.Tensor")

    dtype = type(1.0 + x[0, 0] + y[0, 0])

    xmean = x.mean(axis=1, dtype=dtype)
    ymean = y.mean(axis=1, dtype=dtype)

    # By using `astype(dtype)`, we ensure that the intermediate calculations
    # use at least 64 bit floating point.
    xm = x.astype(dtype) - xmean[:, np.newaxis]
    ym = y.astype(dtype) - ymean[:, np.newaxis]

    # Unlike np.linalg.norm or the expression sqrt((xm*xm).sum()),
    # scipy.linalg.norm(xm) does not overflow if xm is, for example,
    # [-5e210, 5e210, 3e200, -3e200]
    normxm = norm(xm, axis=1)
    normym = norm(ym, axis=1)

    r = np.sum(
        (xm / normxm[:, np.newaxis]) * (ym / normym[:, np.newaxis]), axis=1
    )

    return r











import numpy as np
from scipy.spatial.transform import Rotation as R

# 定义绕每个轴的旋转角度（以度为单位）



def my_3D_rotate(angles):
    # 将角度转换为弧度
    angles_rad = np.radians(angles)
    # 生成旋转对象
    rotation = R.from_euler('xyz', angles_rad)
    # 获取旋转矩阵
    rotation_matrix = rotation.as_matrix()
    # print("旋转矩阵：\n", rotation_matrix)
    return rotation_matrix

angles = [30, 45, 60]  # x轴, y轴, z轴的旋转角度
my_3D_rotate(angles)