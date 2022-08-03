import os
import numpy as np
import nibabel as nib
import random
import torch
from torch.utils import data
import scipy.io as scio


class DataSet(data.Dataset):
    def __init__(self, root, list_path, transform=None):
        super(DataSet,self).__init__()
        self.root = root
        self.list_path = list_path
        ##self.Mask = mask  ## subsampling mask; file 'Real_Mask_Acc4_forTraining.mat' in current folder
        self.img_ids = []
        ## get the number of files. 
        # self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        # print(self.img_ids)
        ## get all fil names, preparation for get_item. 
        ## for example, we have two files: 
        ## 102-field.nii for input, and 102-phantom for label; 
        ## then image id is 102, and then we can use string operation
        ## to get the full name of the input and label files. 
        self.files = []
        for name in self.img_ids:
            label_file = self.root + ("/3shot_patches/Training_patch_%s.nii" % name)
            self.files.append({
                "label": label_file,
                "name": name
            })
        ## sprint(self.files)

    def __len__(self):
        return len(self.files)
 
 
    def __getitem__(self, index):
        datafiles = self.files[index]
 
        '''load the datas'''
        name = datafiles["name"]
        ## nifti read codes. 
        nibLabel = nib.load(datafiles["label"])
        label = nibLabel.get_data()    

        label = np.array(label)
        
        label = torch.from_numpy(label)

        ## convert the image data to torch.tesors and return. 
        image_r = label[:,:,0:1000]
        image_i = label[:,:,1000:2000]
        # image_r = torch.from_numpy(image_r) 
        # image_i = torch.from_numpy(image_i) 
        t1 = label[:,:,2000]
        t2 = label[:,:,2001]
        b0 = label[:,:,2002]

        t1 = torch.reshape(t1,[4096,1])
        t2 = torch.reshape(t2,[4096,1])
        b0 = torch.reshape(b0,[4096,1])


        image_r = torch.reshape(image_r,[4096,1000])
        image_i = torch.reshape(image_r,[4096,1000])
        #image_r = image_r.permute(2, 0, 1)
        #image_i = image_i.permute(2, 0, 1)

        #t1 = torch.unsqueeze(t1, dim = 0)
        #t2 = torch.unsqueeze(t2, dim = 0)
        #b0 = torch.unsqueeze(b0, dim = 0)

        image_r = image_r.float()
        image_i = image_i.float()
        t1 = t1.float()
        t2 = t2.float()
        b0 = b0.float()

        return image_r, image_i, t1, t2, b0, name
 
## before formal usage, test the validation of data loader. 
if __name__ == '__main__':
    DATA_DIRECTORY = '..'
    DATA_LIST_PATH = './test_IDs.txt'
    Batch_size = 5
    dst = DataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print(dst.__len__())
    # just for test,  so the mean is (0,0,0) to show the original images.
    # But when we are training a model, the mean should have another value
    # test code on personal computer: 
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=False)
    for i, Data in enumerate(trainloader):
        imgs, labels, names = Data
        if i%10 == 0:
            print(i)
            print(names)
            print(imgs.size())
            print(labels.size())
