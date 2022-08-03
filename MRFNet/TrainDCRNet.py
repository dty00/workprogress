################### train DCRNet #####################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
from TrainingDataLoad import *
from MRFNet import * 
 
#########  Section 1: DataSet Load #############
def DataLoad(Batch_size):
    DATA_DIRECTORY = '/scratch/itee/uqtding1/trainingdata/'
    DATA_LIST_PATH = '/scratch/itee/uqtding1/trainingdata/3shot_net/test_IDsnew.txt'
    
    dst = DataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print('dataLength: %d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=True, drop_last = True)
    return trainloader

def SaveNet(dcrnet, epo, enSave = False):
    print('save results')
    #### save the
    if enSave:
        pass
    else:
        torch.save(dcrnet.state_dict(), './3shot_MRFNet_final.pth')
        torch.save(dcrnet.state_dict(), ("3shot_MRFNet_%s.pth" % epo))

def TrainNet(dcrnet, LR = 0.001, Batchsize = 32, Epoches = 400 , useGPU = True):
    print('DeepResNet')
    print('DataLoad')
    trainloader = DataLoad(Batchsize)
    print('Dataload Ends')

    print('Training Begins')
    criterion = nn.MSELoss()
    optimizer2 = optim.Adam(dcrnet.parameters())
    scheduler2 = LS.MultiStepLR(optimizer2, milestones = [50,100,150,200,250,300,350], gamma = 0.3)

    ## start the timer. 
    time_start=time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dcrnet = nn.DataParallel(dcrnet)
            dcrnet.to(device)

            for epoch in range(1, Epoches + 1):
                
                if epoch % 100 == 0:
                    SaveNet(dcrnet, epoch, enSave = False)

                acc_loss = 0.0
                for i, data in enumerate(trainloader):
                    image_r, image_i, T1s, T2s, B0s, Name = data
                    image_r = image_r.to(device)
                    image_i = image_i.to(device)
                    T1s = T1s.to(device)
                    T2s = T2s.to(device)
                    B0s = B0s.to(device)
                    #print(image_i.size())
                    
                    ## zero the gradient buffers 
                    optimizer2.zero_grad()
                    ## forward: 
                    
                    pred_t1, pred_t2, pred_b0 = dcrnet(image_r, image_i)

                    pred_t1 = torch.unsqueeze(pred_t1, dim =0)
                    pred_t2 = torch.unsqueeze(pred_t2, dim =0)
                    pred_b0 = torch.unsqueeze(pred_b0, dim =0)
                    #print("pred_t1 size",pred_t1.size())
                    #print('t1 size',T1s.size())
                    loss1 = criterion(pred_t1 / 4e3, T1s / 4e3)
                    loss2 = criterion(pred_t2 / 200, T2s / 200)
                    loss3 = criterion(pred_b0 / 200, B0s / 200)

                    loss = loss1+loss2+loss3

                    # loss = criterion(pred, label)

                    loss.backward()
                    ##
                    optimizer2.step()
                    optimizer2.zero_grad()
                    
                    ## print statistical information 
                    ## print every 20 mini-batch size
                    if i % 100 == 0:
                        acc_loss1 = loss1.item()   
                        acc_loss2 = loss2.item() 
                        acc_loss3 = loss3.item()  
                        time_end=time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss_final: %f \n lr2: %f, used time: %d s' %
                            (epoch, i + 1, acc_loss1+acc_loss2+acc_loss3, optimizer2.param_groups[0]['lr'], time_end - time_start))   
                scheduler2.step()
        else:
            pass
            print('No Cuda Device!')
            quit()        
    print('Training Ends')
    SaveNet(dcrnet, Epoches, enSave = False)

if __name__ == '__main__':
    ## data load
    ## create network 
    dcrnet = MRFNet(3)
    dcrnet.apply(weights_init)
    print(dcrnet.state_dict)
    print(get_parameter_number(dcrnet))
    ## train network
    TrainNet(dcrnet, LR = 0.0001, Batchsize = 20, Epoches = 400, useGPU = True)
    torch.save(dcrnet, 'model_fcntest.pkl')


