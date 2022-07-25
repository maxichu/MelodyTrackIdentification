import torch
from torch import dropout, nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BigBirdForSequenceClassification

# sys.path.append('../SparseConvNet/')
# import sparseconvnet as scn
# from sparseconvnet.activations import ReLU


# def make_sparse_layers(cIn, layers, kernel=4, dimension=2, pre_conv=True):
#     '''
#     adpot from https://github.com/facebookresearch/SparseConvNet
#     '''
#     m = scn.Sequential()
#     if pre_conv:
#         m.add(scn.Convolution(dimension, 1, cIn, kernel, 1, False))
#         m.add(scn.ReLU())
#     current_cIn = cIn
#     def residual(nIn, nOut, stride):
#         if stride > 1:
#             return scn.Convolution(dimension, nIn, nOut, kernel, stride, False)
#         elif nIn != nOut:
#             return scn.NetworkInNetwork(nIn, nOut, False)
#         else:
#             return scn.Identity()

#     for cOut, reps, stride in layers:
#         for rep in range(reps):
#             if rep == 0:

#                 main_path = scn.Sequential(
#                     scn.SubmanifoldConvolution(dimension, current_cIn, cOut, kernel, False),
#                     scn.ReLU(),
#                     scn.SubmanifoldConvolution(dimension, cOut, cOut,  kernel, False),
#                     #scn.ReLU()
#                 )
#                 skip_path = residual(current_cIn, cOut, stride)
#                 # concatTable return [math_path_values, skip_path_value] 
                
#                 m.add(scn.ConcatTable(main_path, skip_path)) 
                 
#             else:
#                 main_path = scn.Sequential(
#                     scn.ReLU(),
#                     scn.SubmanifoldConvolution(dimension, cOut, cOut,  kernel, False),
#                     scn.ReLU(),
#                     scn.SubmanifoldConvolution(dimension, cOut, cOut, kernel, False)
#                 )
#                 skip_path = scn.Identity()
#                 m.add(scn.ConcatTable(main_path, skip_path)) 
                   
#             # AddTable simply add all paths in input together
#             m.add(scn.AddTable()) 

#         # assign current out channel number to next layer's input number
#         current_cIn = cOut
#         m.add(scn.ReLU())
#     m.add(scn.SparseToDense(dimension,current_cIn))
#     return m

# class SparseResNet(nn.Module):
#     def __init__(self, option=1):
#         nn.Module.__init__(self)
#         self.opt = int(option)
        
#         self.sparse_res = make_sparse_layers(16, [[32,1,1]], kernel=3)
#         self.linear_in = 3200
#         self.sparse_out = torch.LongTensor([10, 10])
#         #else throw exception

#         self.dropout1 = nn.Dropout(p=0.5)
#         self.dropout2 = nn.Dropout(p=0.3)
#         self.relu = nn.ReLU()
#         self.flatten = nn.Flatten()
#         self.linear1 = nn.Linear(self.linear_in,512)
#         self.linear2 = nn.Linear(512, 16)
#         self.inputSpatialSize = self.sparse_res.input_spatial_size(self.sparse_out)
#         self.input_layer = scn.InputLayer(2, self.inputSpatialSize)
    
#     def forward(self,x):
#         x = self.input_layer(x) # x = [locations,features,bs]
        
#         x = self.sparse_res(x)
#         x = self.flatten(x)
#         x = self.dropout1(x)
#         #x = self.linear1(x)
#         x = self.relu(self.linear1(x))
#         x = self.dropout2(x)
#         x = self.linear2(x)
#         return x

# class SparseConvNet(nn.Module):
#     def __init__(self,option='scn'):
#         nn.Module.__init__(self)

#         self.opt = option
#         if self.opt == 'scn':
#             self.sparse_conv = scn.Sequential(
#             scn.Convolution(2, 1, 16, 4, 1, False), #dimension, nIn, nOut, filter_size, filter_stride, bias
#             scn.ReLU(),
#             scn.Convolution(2, 16, 64, 4, 1, False),
#             scn.ReLU(),
#             scn.SparseToDense(2, 64) #dimension(coords), out_channel
#         )

#         elif self.opt == 'sscn':        
#             self.sparse_conv = scn.Sequential(
#                 scn.SubmanifoldConvolution(2, 1, 16, 4, False), #dimension, nIn, nOut, filter_size, bias
#                 scn.ReLU(),
#                 scn.SubmanifoldConvolution(2, 16, 32, 4, False), #dimension, nIn, nOut, filter_size, bias
#                 scn.ReLU(),
#                 scn.SparseToDense(2, 32) #dimension(coords), out_channel
#             )
#         #else throw exception

#         self.dropout1 = nn.Dropout(p=0.5)
#         self.dropout2 = nn.Dropout(p=0.3)
#         self.relu = nn.ReLU()
#         self.flatten = nn.Flatten()
#         self.linear1 = nn.Linear(6400,512)
#         self.linear2 = nn.Linear(512, 16)
#         self.inputSpatialSize = self.sparse_conv.input_spatial_size(torch.LongTensor([10, 10]))
#         self.input_layer = scn.InputLayer(2, self.inputSpatialSize)
    
#     def forward(self,x):
#         x, m = x
#         x = self.input_layer(x) # x = [locations,features,]
#         x = self.sparse_conv(x)
#         x = self.flatten(x)
#         x = self.dropout1(x)
#         x = self.relu(self.linear1(x))
#         x = self.dropout2(x)
#         x = self.linear2(x)
#         return x



class ConvNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.cnn = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=4,padding=2,bias=False),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=4,padding=1,bias=False),
            nn.ReLU()
        )
        self.linear1 = nn.Linear(4096,512)
        self.linear2 = nn.Linear(512,16)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self,x):
        m = x.detach().clone()
        m[m!=0] = 1
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.relu(self.linear1(x))
        x = self.dropout2(x)
        x = self.linear2(x)

        return x


class BasicConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(  
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        )
        #self.bn = nn.BatchNorm2d(out_channel) if bn else None
        # self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return self.conv(x)
        # return self.bn(self.conv(x))
        #h = self.bn(h) if self.bn is not None else h

def make_layers(in_shape, layers):
    skips = []
    convs = []
    c, w, _ = in_shape
    for layer in layers:
        in_channel, out_channel, kernel, stride, padding = layer
        skips.append(nn.Identity() if in_channel == out_channel \
            else nn.Conv2d(in_channel, out_channel ,kernel_size=1))
        convs.append(BasicConvBlock(in_channel, out_channel, kernel, 1, padding))
        
        w = (w - kernel + 2*padding) // stride + 1
        c = out_channel
        
    return nn.ModuleList(skips), nn.ModuleList(convs), c*w*w

layers_opt = {
    0: [[16,64,3,1,1]],
    1: [[16,64,3,1,1],[64,64,3,1,1]],
    2: [[16,32,3,1,1],[32,64,3,1,1],[64,64,3,1,1]]
}


class ResConvNet(nn.Module):
    def __init__(self, resblock_opt=1, in_shape=(1,16,16), out_shape=16, linear_head=None, pretrained=None, dropout1=0.3, dropout2=0.1): #, masking=False): 
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        #self.masking = masking
        self.layers = layers_opt[int(resblock_opt)]
        self.conv0 = nn.Conv2d(1,16,3,padding=1,bias=False)
        self.skips, self.blocks, self.in_feature = make_layers(in_shape, self.layers)
        if pretrained or not linear_head:
            self.avg_pool = nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
            self.flatten = nn.Flatten()
            self.dropout1 = nn.Dropout(p=dropout1)
            self.linear1 = nn.Linear(self.in_feature // 4, 512, bias=False)
            self.dropout2 = nn.Dropout(p=dropout2)
            self.linear2 = nn.Linear(512, out_shape, bias=False)
            self.linear_head = linear_head
        else: 
            self.linear_head = nn.Sequential(
                nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(self.in_feature // 4, 512, bias=False),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, out_shape, bias=False)
            ) if not linear_head else linear_head
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()

    def forward(self, input, plain_cnn=True):
        x, _, _, m = input # do not change
        x = self.relu(self.conv0(x))
        for i in range(len(self.layers)):
            x = self.relu(self.skips[i](x) + self.blocks[i](x)) # skip path
        # if self.masking: # not working 
        #     x = x[:,:,7,:].sum(1).reshape((-1,16))
        # x = self.linear_head(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        if self.linear_head:
            x = self.linear_head(x)
            # x[m==0] = -float("inf")
            # x = self.sm(x)
        elif plain_cnn:
            x = self.relu(self.linear1(x))
            x = self.dropout2(x)
            x = self.linear2(x)
            # x[m==0] = -float("inf")
            # x = self.sm(x)
        else:
            x = self.linear1(x)
            # x = self.sm(x)
        return x
    
class MelodyNet(nn.Module):
    def __init__(self, resblock_opt = 1, in_shape=(1,16,16), cnn_out_dim=512, 
    n_class=16, hidden_dim=512, rnn_layer=1, rnn_bi=True, bias=True, dropout=0.3, 
    rnn_dropout=0.15, seq_head='crnn', bigbirdconfig=None, batch_first=True, pretrained=None, device='cuda:0'):
        super().__init__()
        if pretrained:
            if pretrained.split('.')[-1] == 'pt':
                checkpoint = torch.load(pretrained,map_location=device)
                self.cnn = ResConvNet(resblock_opt=1,pretrained=pretrained)
                self.cnn.load_state_dict(checkpoint['state_dict'])
            elif pretrained.split('.')[-1] == 'pth':
                self.cnn = torch.jit.load(pretrained)
            for param in self.cnn.parameters():
                param.requires_grad = False
            cnn_out_dim = 512 # n_class
        else:
            self.cnn_linhead = nn.Sequential(
                nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, cnn_out_dim, bias=False) # to do: reconstruct code
            )
            self.cnn = ResConvNet(resblock_opt,in_shape,linear_head=self.cnn_linhead)
        rnn_dropout = rnn_dropout if rnn_layer > 1 else 0
        self.seq_head = seq_head
        self.sm = nn.Softmax()
        if self.seq_head == 'crnn':
            self.rnn = nn.LSTM(cnn_out_dim,hidden_dim,rnn_layer,bias=bias,
            batch_first=batch_first,bidirectional=True,dropout=rnn_dropout)
            self.dropout = nn.Dropout(p=dropout)
            self.linear_head = nn.Linear(hidden_dim,n_class)
        elif self.seq_head == 'bigbird':
            self.transformer = BigBirdForSequenceClassification(bigbirdconfig)
            

    def forward(self, input):
        x, _, l, m = input
        N, T, C, H, W = x.size() 
        x = x.view(N*T, C, H, W)
        x = self.cnn([x,None,None, m],plain_cnn=False) 
        x = x.view(N, T, -1)
        if self.seq_head == 'crnn':
            x = pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
            _, (x, _) = self.rnn(x) 
            x = x.mean(axis=0)
            x = self.dropout(x)
            x = self.linear_head(x) 
        elif self.seq_head == 'bigbird':
            x = self.transformer(inputs_embeds=x,attention_mask=l).logits
        # x = x.clone()
        # print(x)
        # x[m==0] = -float("inf")
        # print(x)
        # print(m)
        # return self.sm(x)  
        return x
    
class MelodyRNN(nn.Module):
    def __init__(self, hidden_dim=512, layer=1, n_class=16, dropout=0.1, rnn_bi=True):
        super().__init__()
        self.rnn = nn.LSTM(16,hidden_dim,layer,
            batch_first=True,bidirectional=rnn_bi)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_head = nn.Linear(hidden_dim,n_class)
    
    def forward(self, input):
        x, _, _, _ = input

        _, (x, _) = self.rnn(x)
        x = x.mean(0)
        x = self.linear_head(x)
        return x
    

# import torch

# def masked_softmax(x, mask, **kwargs):
#     x_masked = x.clone()
#     x_masked[mask == 0] = -float("inf")

#     return torch.softmax(x_masked, **kwargs)