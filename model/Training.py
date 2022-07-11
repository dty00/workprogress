from cProfile import label
from random import shuffle
from model.MLPNetwork import MLPNet, get_parameter_number
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
from TrainingDataLoad import *
from MRLPNet import * 

def DataLoad(Batch_size):
    DATA_DIRECTORY = "../MRF_cnn/"
    DATA_LIST_PATH = "./test_IDs.txt"

    dst = DataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print('datalength:%d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size=Batch_size, shuffle=True, drop_last =True)

    return trainloader

def SaveNet(mlpnet, epo, enSave = False):
    print('save results')

    if enSave:
        pass
    else:
        torch.save(mlpnet.state_dict(),'./MLPNet_final.pth')
        torch.save(mlpnet.state_dict(),('MLPNet_%s.pth'% epo))

def TrainNet(mlpnet, LR = 0.001, Batchsize = 2, Epoches = 200, useGPU = True):
    print("TestMLPNet")
    print('DataLoad')
    trainloader = DataLoad(Batchsize)
    print('Dataload ends')

    print("training Begins")
    criterion = nn.MSELoss()
    optimizer2 = optim.Adam(mlpnet.parameters())
    scheduler2 = LS.MultiStepLR(optimizer2, milestones = [30,50,80,100,150,200], gamma = 0.3)

    time_start = time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), 'Available GPUs!')
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            mlpnet = nn.DataParallel(mlpnet)
            mlpnet.to(device)

            for epoch in range(1,Epoches+1):
                if epoch % 50 == 0:
                    SaveNet(mlpnet, epoch, enSave=False)
                
                acc_loss = 0.0
                for i, data in enumerate(trainloader):
                    x, label = data
                    label = label.to(device)
                    optimizer2.zero_grad()

                    pred = mlpnet(x)
                    loss = criterion(pred, label)

                    loss.backward()
                    optimizer2.step()
                    optimizer2.zero_grad()

                    if i % 50 ==0:
                        acc_loss1 = loss.item()
                        time_end = time.time()
                        print("OutsideL Eopch : %d, batch: %d, loss_final: %f \n lr2: %f, used time: %d s"%(epoch, i+1, acc_loss1, optimizer2.param_groups[0]['lr'],time_end-time_start))
                scheduler2.step()
        else:
            pass
            print('No Cuda Device!')
            quit()
    print('Training ends')
    SaveNet(mlpnet, Epoches, enSave = False)


if __name__ == '__main__':

    mlpnet = MLPNet()
    mlpnet.apply(weights_init)
    print(mlpnet.state_dict)
    print(get_parameter_number(mlpnet))

    TrainNet(mlpnet, LR = 0.001, Batchsize= 10, Epoches= 200, useGPU=True)