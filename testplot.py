# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy.io
import numpy as np
import math as m
import torch
import torch.nn as nn
imag = scipy.io.loadmat('E:/img_10shots_noSVD.mat')
from torch.utils.data import Dataset, DataLoader

datas = imag['img']
im1 = datas[0]
im2 = datas[1]

reim1 = np.squeeze(im1.reshape(25600,1000,order ='F'))
reim2 = np.squeeze(im2.reshape(25600,1000,order = 'F'))
nim1 = reim1/np.linalg.norm(reim1,axis=1,keepdims=True)
nim2 = reim2/np.linalg.norm(reim2,axis=1,keepdims=True)

real_num1 = np.real(nim1)
imag_num1 = np.imag(nim1)
real_num2 = np.real(nim2)
imag_num2 = np.imag(nim2)

class testDataset(Dataset):
    def __init__(self,xr,xi):
        self.xr = xr
        self.xi = xi

    def __len__(self):
        return len(self.xr)
    
    def __getitem__(self, idx):
        return self.xr[idx],self.xi[idx]

image_data1 = testDataset(real_num1,imag_num1)
image_data2 = testDataset(real_num2,imag_num2)

test_dataloader1 = DataLoader(image_data1, shuffle=False, batch_size=1)
test_dataloader2 = DataLoader(image_data2,shuffle=False,batch_size = 1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


model = torch.load('H:/tding/data/fcn_test.pkl')
model.eval()

real_num1_1d = np.squeeze(real_num1.reshape(25600,1000))
imag_num1_1d = np.squeeze(imag_num1.reshape(25600,1000))

ops1 = []
ops2 = [] 
with torch.no_grad():
    for i, (rea1,img1) in enumerate(test_dataloader1):

        real1 = rea1.to(device,dtype=torch.float)
        imga1 = img1.to(device,dtype=torch.float)
        real1 = torch.reshape(real1,[1000,1])
        imga1 = torch.reshape(imga1,[1000,1])
        #print(real1.shape)
        real1 = torch.unsqueeze(real1, dim = 0)
        #print(real1.shape)
        imga1 = torch.unsqueeze(imga1,dim=0)

        t11,t21,b01 = model(real1,imga1)
        ops1.append([t11.item(),t21.item(),b01.item()])
    for i,(rea2,img2) in enumerate(test_dataloader2):
        real2 = rea2.to(device,dtype=torch.float)
        imga2 = img2.to(device,dtype=torch.float)
        real2 = torch.reshape(real2,[1000,1])
        imga2 = torch.reshape(imga2,[1000,1])
        
        real2 = torch.unsqueeze(real2,dim=0)
        imga2 = torch.unsqueeze(imga2,dim=0)
        t12,t22,b02 = model(real2,imga2)
        ops2.append([t12.item(),t22.item(),b02.item()])

# res1 = torch.cat(ops1,dim=0)
# res2 = torch.cat(ops2,dim=0)
res1 = torch.FloatTensor(ops1)
res2 = torch.FloatTensor(ops2)   


"save to matlab matrix"

res1 = res1.cpu()
res2 = res2.cpu()
result1 = res1.numpy()
result2 = res2.numpy()
scipy.io.savemat('H:/tding/data/invivo1.mat',{'recons1':result1})
scipy.io.savemat('H:/tding/data/invivo2.mat',{'recons2':result2})