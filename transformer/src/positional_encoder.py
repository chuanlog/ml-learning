import math

import torch
from torch import nn

'''
位置编码器
'''
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        '''
        d_model: 模型中向量的维度
        max_len: 最大维度
        '''
        super().__init__()
        position=torch.arange(0, max_len).unsqueeze(1) # 变成[[0] [1] [2] [3] .....]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe=torch.zeros(1,max_len,d_model)





