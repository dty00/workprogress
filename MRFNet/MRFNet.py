import torch
import torch.nn as nn 
import torch.nn.functional as F

class MRFNet(nn.Module):
    def __init__(self, EncodingDepth = 8, initial_num_layers = 2048):
        super(MRFNet, self).__init__()

        self.EncodingDepth = EncodingDepth

        self.init1 = CConv1d_BN_RELU(1000, initial_num_layers, 1, 0) 

        self.midLayers = []
        temp = list(range(1, EncodingDepth + 1))
        for encodingLayer in temp:
            inl = initial_num_layers // (2 ** (encodingLayer - 1))
            outl = initial_num_layers // (2 ** encodingLayer)
            self.midLayers.append(CConv1d_BN_RELU(inl, outl, 1, pad = 0))
        self.midLayers = nn.ModuleList(self.midLayers)
                        
        self.final = nn.Linear(outl * 2, 1)

    def forward(self, x_r, x_i):
        #print(x_r.shape)
        x_r = x_r.permute(0, 2, 1)  ## nb * 1000 * 1
        x_i = x_i.permute(0, 2, 1) 

        x_r, x_i  = self.init1(x_r, x_i)

        temp = list(range(1, self.EncodingDepth + 1))
        for encodingLayer in temp:
            temp_conv = self.midLayers[encodingLayer - 1]
            x_r, x_i = temp_conv(x_r, x_i)

        x_r = x_r.permute(0, 2, 1)
        x_i = x_i.permute(0, 2, 1) 

        x = torch.cat([x_r, x_i], dim = -1)

        #print(x.size())
        x = self.final(x)
        t1 = x[0]
        t2 = x[1]
        b0 = x[2]
        return t1, t2, b0



class Basic_block(nn.Module):
    def __init__(self, num_in, num_out):
        super(Basic_block, self).__init__()
        self.cconv1 = CConv1d_BN_RELU(num_in, num_out)
        self.cconv2 = CConv1d_BN_RELU(num_out, num_out)

    def forward(self, x_r, x_i):
        INPUT_r = x_r
        INPUT_i = x_i
        x_r, x_i = self.cconv1(x_r, x_i)
        x_r = x_r + INPUT_r
        x_i = x_i + INPUT_i
        x_r, x_i = self.cconv2(x_r, x_i)
        return x_r, x_i

## complex convolution; 
class CConv1d_BN_RELU(nn.Module):
    def __init__(self, num_in, num_out, ks = 3, pad = 1):
        super(CConv1d_BN_RELU, self).__init__()
        self.conv_r = nn.Conv1d(num_in, num_out, ks, padding= pad)
        self.conv_i = nn.Conv1d(num_in, num_out, ks, padding= pad)
        # self.bn_r = nn.BatchNorm1d(num_out)
        # self.bn_i = nn.BatchNorm1d(num_out)
        self.relu_r = nn.ReLU(inplace = True)
        self.relu_i = nn.ReLU(inplace = True)

    def forward(self, x_r, x_i):
        x_rr = self.conv_r(x_r)
        x_ri = self.conv_i(x_r)
        x_ir = self.conv_r(x_i)
        x_ii = self.conv_i(x_i)
        x_r = x_rr - x_ii 
        x_i = x_ri + x_ir
        # x_r = self.bn_r(x_r)
        # x_i = self.bn_i(x_i)
        x_r = self.relu_r(x_r)
        x_i = self.relu_i(x_i)
        return x_r, x_i


## complex convolution; 
class CConv1d(nn.Module):
    def __init__(self, num_in, num_out, ks = 1, pad = 0, bs = True):
        super(CConv1d, self).__init__()
        self.conv_r = nn.Conv1d(num_in, num_out, ks, bias = bs, padding= pad)
        self.conv_i = nn.Conv1d(num_in, num_out, ks, bias = bs, padding= pad)
        
    def forward(self, x_r, x_i):
        x_rr = self.conv_r(x_r)
        x_ri = self.conv_i(x_r)
        x_ir = self.conv_r(x_i)
        x_ii = self.conv_i(x_i)
        x_r = x_rr - x_ii 
        x_i = x_ri + x_ir

        return x_r, x_i

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
# if __name__ == '__main__':
#     mrsnet = MRFNet()
#     mrsnet.apply(weights_init)
#     print(mrsnet.state_dict)
#     print(get_parameter_number(mrsnet))
#     x_r = torch.randn(2,1,1000, dtype=torch.float)
#     x_i = torch.randn(2,1,1000, dtype=torch.float)
#     print('input' + str(x_r.size()))
#     print(x_r.dtype)
#     y = mrsnet(x_r, x_i)
#     print('output'+str(y.size()))
