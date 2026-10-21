# 手写一遍数字识别的网络训练
import matplotlib
from matplotlib import image
from numpy.typing import NDArray
import torch
import torch.nn as nn 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# 工具函数
def load_idx_images(path: str | Path) -> NDArray[np.uint8]:
    '''
    加载idx格式的图片数据集
    '''
    with open(path, 'rb') as f: # 打开文件
        header=np.frombuffer(f.read(16), dtype=np.dtype('>u4')) # 读取头部信息,
        magic, n_items, n_rows, n_cols = header # 解析头部信息（魔数，图片数量，图片高度，图片宽度）
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n_items, n_rows, n_cols) # 读取图片数据
    return images

def load_idx_labels(path: str | Path) -> NDArray[np.uint8]:
    '''
    加载idx格式的标签数据集
    '''
    with open(path, 'rb') as f: # 打开文件
        header=np.frombuffer(f.read(8), dtype=np.dtype('>u4')) # 读取头部信息,
        magic, n_items = header # 解析头部信息（魔数，标签数量）
        labels = np.frombuffer(f.read(), dtype=np.uint8) # 读取标签数据
    return labels
# 加载数据集
data_dir=Path(__file__).parent / 'data' / 'MNIST' / 'raw'  # base路径
images = load_idx_images(data_dir / 'train-images-idx3-ubyte')
labels = load_idx_labels(data_dir / 'train-labels-idx1-ubyte')


# 看一张数据集
plt.imshow(images[0],'gray')
plt.show()
print(images.size)

# 把数据集变成
