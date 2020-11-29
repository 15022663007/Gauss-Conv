# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 01:47:29 2020

@author: Administrator
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[0.03797616, 0.044863533, 0.03797616],
                  [0.044863533, 0.053, 0.044863533],
                  [0.03797616, 0.044863533, 0.03797616]]
        k = 1
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0) * k

        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x,kk = 0.1):
        
        x = F.conv2d(x, self.weight, stride = (1,1), padding=(1,1)) + 0.01 * kk
        x = F.conv2d(x, self.weight, stride = (1,1), padding=(1,1)) + 0.1 * kk
        x = F.conv2d(x, self.weight, stride = (1,1), padding=(1,1)) - 0.05 * kk
        x = F.conv2d(x, self.weight, stride = (1,1), padding=(1,1)) + 0.2 * kk
        x = F.conv2d(x, self.weight, stride = (1,1), padding=(1,1)) + 0.01 * kk
        x = F.conv2d(x, self.weight, stride = (1,1), padding=(1,1)) - 0.02 * kk
        x = F.conv2d(x, self.weight, stride = (1,1), padding=(1,1)) + 0.2 * kk
        x = F.conv2d(x, self.weight, stride = (1,1), padding=(1,1)) + 0.1 * kk
        x = F.conv2d(x, self.weight, stride = (1,1), padding=(1,1)) - 0.01 * kk
        x = F.conv2d(x, self.weight, stride = (1,1), padding=(1,1)) - 0.05 * kk

        return x


def normalization(data):
    _range = np.max(data) - np.min(data)
    print('range:',_range)
    return (data - np.min(data)) / _range

    

tensor_input = torch.ones([1,1,100,100])
model = GaussianBlur()
out = model(tensor_input)

out_numpy = out.cpu().numpy()
out_numpy = out_numpy.squeeze()

out_numpy = normalization(out_numpy) * 255

print(out_numpy)

out_numpy = np.expand_dims(out_numpy,2)
print('output shape:',out_numpy.shape)


cv2.imwrite("./out4.png", out_numpy,[int(cv2.IMWRITE_JPEG_QUALITY), 0])
#第一个参数是保存的路径及文件名，第二个是图像矩阵。
#第三个参数针对特定的格式： 
#对于JPEG，其表示的是图像的质量，用0-100的整数表示，默认为95。
#对于PNG，第三个参数表示的是压缩级别。cv2.IMWRITE_PNG_COMPRESSION，从0到9,压缩级别越高，图像尺寸越小。默认级别为3





