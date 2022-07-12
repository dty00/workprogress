import os
import numpy as np
import random
import torch
from torch.utils import data
import scipy.io as scio

class DataSet(data.Dataset):
    def __init__(self, root,list_path,transform=None):
        super(DataSet,self).__init__()
        self.root = root
        self.list_path = list_path
        self.img_ids = []

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []

        for name in self.img_ids:
            lfs_file = self.root+('/lfs_training/lfs_patch_%s.mat'%name)
            chi_file = self.root+('/mlp_training/chi_patch_%s.mat'%name)
            self.files.append({
                "lfs":lfs_file,
                "chi":chi_file
                "name":name 
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        """load the datas"""
        name = datafiles['name']

        