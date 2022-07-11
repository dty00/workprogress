import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(50,200)
        self.fc2 = nn.Linear(200,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,32)
        self.fc5 = nn.Linear(32,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)

        return x


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(m.bias)   
    if isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)   

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


    
#################### For Code Test ##################################
## before running the training codes, verify the network architecture. 
if __name__ == '__main__':
    mlpnet = MLPNet()
    mlpnet.apply(weights_init)
    print(mlpnet.state_dict)
    print(get_parameter_number(mlpnet))
    x_r = torch.randn(200,50, dtype=torch.float)
    print('input' + str(x_r.size()))
    print(x_r.dtype)
    y = mlpnet(x_r)
    print('output'+str(y.size()))
        
