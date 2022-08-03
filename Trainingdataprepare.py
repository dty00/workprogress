import os
import numpy as np
import nibabel as nib
import random
import torch
from torch.utils import data
import scipy.io as scio

DATA_DIRECTORY = '/mnt/h/tding/data'
DATA_LIST_PATH = '/mnt/h/tding/data/test_IDs.txt'

img_ids = []
img_ids = [i_id.strip() for i_id in open(DATA_LIST_PATH)]

files = []
for name in img_ids:
    label_file = '/mnt/h/tding/data/syn_10shot/TrainObj_12_slice_%s.nii'% name
    files.append({"label":label_file,"name":name})

concatdata2 = []
for i in files:
    nibLabel = nib.load(i['label'])
    label = nibLabel.get_fdata()
    label = torch.from_numpy(label)
    label = torch.reshape(label,[25600,2003])

    if i['name'] =='1':
        concatdata2 = label
    elif int(i['name']) %10 == 0:
        torch.save(concatdata2,'/mnt/h/tding/data/syn_10shot_patch/slice12_%s.pt'%i['name'])
        concatdata2 = label
    else:
        print(i['name'],'/160 remaining')
        concatdata2 = torch.cat([concatdata2,label], dim = 0)

if __name__ =='__main__':
    pass