#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import time
import numpy as np
import pandas as pd
import cv2
import random

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from datetime import datetime

import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
from torch.utils.data import DataLoader

from sklearn.preprocessing import binarize
# import enviroments
# import torchsummary

import torchvision
import torchvision.transforms as trns
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


from tqdm import tqdm 


# In[8]:


def img_preprocessing(img):
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite('output1.jpg', image)
#     image = cv2.imread(path_data)
    image = cv2.resize(image, (140, 48), interpolation=cv2.INTER_CUBIC)
    dst = cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)
    ret, thresh = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY_INV)
    image = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    image_res = cv2.resize(image,(140,48),interpolation = cv2.INTER_CUBIC)
    image_res[:,10:135] = 0
    imagedata = np.where(image_res == 255)
    # plt.scatter(imagedata[1],47 - imagedata[0], s = 100 , c = 'red', label = 'Cluter')
    # plt.ylim(ymin = 0)
    # plt.ylim(ymax = 47)
    # plt.show()
    X = np.array([imagedata[1]])
    Y = 47 - imagedata[0]
    poly_reg = PolynomialFeatures(degree = 2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_,Y)
    X2 = np.array([[i for i in range(0,139)]])
    X2_ = poly_reg.fit_transform(X2.T)
    # %pylab inline
    # plt.scatter(X,Y, color = 'black')
    # plt.ylim(ymin = 0)
    # plt.ylim(ymax = 47)
    # plt.plot(X2.T,regr.predict(X2_),color = 'blue')
    # plt.show()
    newimg = cv2.cvtColor(thresh , cv2.COLOR_BGR2GRAY)
    for ele in np.column_stack([regr.predict(X2_).round(0),X2[0],]):
        pos = 47 - int(ele[0])
        newimg[pos-3:pos+3,int(ele[1])] = 255 - newimg[pos-3:pos+3,int(ele[1])]
        
    cv2.imwrite('output2.jpg', newimg)
    return newimg


# In[9]:


def decoding(onehot_code):
    digit1 = torch.argmax(onehot_code[0],axis= 1).cpu().item()
    digit2 = torch.argmax(onehot_code[1],axis= 1).cpu().item()
    digit3 = torch.argmax(onehot_code[2],axis= 1).cpu().item()
    digit4 = torch.argmax(onehot_code[3],axis= 1).cpu().item()


    CAPTCHA_DICT = {0: '2',1: '3',2: '4',3: '5',4: '7',5: '9',6: 'A',7: 'C',8: 'F',9: 'H',10: 'K',11: 'M',12: 'N',13: 'P',14: 'Q',15: 'R',16: 'T',17: 'Y',18: 'Z'}

    str1 = CAPTCHA_DICT[digit1]
    str2 = CAPTCHA_DICT[digit2]
    str3 = CAPTCHA_DICT[digit3]
    str4 = CAPTCHA_DICT[digit4]
    
    return(str1+str2+str3+str4)


# In[21]:


def test(img ,model, device):
    """!!! Useless !!!
    """


    
    data = img_preprocessing(img)
    
    
    img2 = np.zeros( (3, np.array(data).shape[0], np.array(data).shape[1] ) )
    img2[0,:,:] = data[:,:] # same value in each channel
    img2[1,:,:] = data[:,:]
    img2[2,:,:] = data[:,:]
    
    img2 = torch.from_numpy(img2)
    img2 = img2.expand(1,-1,-1,-1)
    
    
    with torch.no_grad():
        img2 = img2.to(device)
        img2 = img2.float()
#         plt.imshow(data)
        score = model(img2)
        score = (score)
        score = decoding(score)
        return score



class MyAlexNet(nn.Module):
  def __init__(self):
    super(MyAlexNet, self).__init__()
    
    pretrained = torchvision.models.resnet18(pretrained=False)
    num = pretrained.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    pretrained.fc = nn.Sequential(
                   nn.Linear(num, num),
                   nn.ReLU(inplace=True),
                   nn.Linear(num, 512))
    
    
    self.pretrained = pretrained

    self.digit1 = nn.Linear(512, 19)
    self.digit2 = nn.Linear(512, 19)
    self.digit3 = nn.Linear(512, 19)
    self.digit4 = nn.Linear(512, 19)
  
  def forward(self, x):
    x = self.pretrained(x)
    digit1 = torch.nn.functional.softmax(self.digit1(x), dim=1)
    digit2 = torch.nn.functional.softmax(self.digit2(x), dim=1)
    digit3 = torch.nn.functional.softmax(self.digit3(x), dim=1)
    digit4 = torch.nn.functional.softmax(self.digit4(x), dim=1)
    return digit1, digit2, digit3, digit4


def get_captcha_code(image):
    device = torch.device("cpu")
    model_path  = './thsr_ticket/Resnet/Resnet.pth'
    model = MyAlexNet() 
    model.load_state_dict(torch.load(model_path, map_location = device))
#     model = torch.load(model_path, map_location = device)
#     print(model)
    model.to(device)
        # 把模型设为验证模式
    model.eval()
    return test(image ,model, device)


# In[23]:




# import time
# def tic():
#     globals()['tt'] = time.time()
# def toc():
# #     print('Elapsed time: %.8f seconds' % (time.time()-globals()['tt']))
#     return round((time.time()-globals()['tt']),4)


# In[ ]:


# if __name__ == "__main__":
#     delt_time =[]
#     for i in range(1,10):
#         tic()
#         path_data = './captcha/' + str(random.randint(1,10000)) + '.jpg'
#         model_path  = './thsr_ticket/Resnet/CNN_20210922-115002_24_0.999268.pth'
#         image = cv2.imread(path_data)
#         test(image ,model, model_path , device)
#         delt_time.append(toc())

#     np.mean(delt_time)
    



