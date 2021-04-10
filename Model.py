import torch.nn as nn
import torch
from functions import ReverseLayerF
from torch.autograd import Variable


l1=640;l2=320;d=0.25
class msdan(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size,in_channels,out_channels, kernel_size):
        super(msdan, self).__init__()

        self.bgru1 = nn.GRU(input_size, hidden_size, layer_size, batch_first=True,bidirectional=True)

        self.cnn = nn.Sequential()
        self.cnn.add_module('conv1',nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=0))
        self.cnn.add_module('bn2', nn.BatchNorm1d(out_channels))
        self.cnn.add_module('pool3',nn.MaxPool1d(2))
        self.cnn.add_module('relu4', nn.ReLU())
        self.cnn.add_module('conv5',nn.Conv1d(out_channels, out_channels*2, kernel_size, stride=2, padding=0))
        self.cnn.add_module('bn6', nn.BatchNorm1d(out_channels*2))
        self.cnn.add_module('pool7',nn.MaxPool1d(2))
        self.cnn.add_module('relu8', nn.ReLU())
        self.cnn.add_module('conv9',nn.Conv1d(out_channels*2, out_channels*4, kernel_size, stride=2, padding=0))
        self.cnn.add_module('bn10', nn.BatchNorm1d(out_channels*4))
        self.cnn.add_module('pool11',nn.MaxPool1d(2))
        self.cnn.add_module('relu12', nn.ReLU())

        self.specific1 = nn.Sequential()
        self.specific1.add_module('fc1', nn.Linear(l1, l2))#specific feature of source domain 1
        self.specific1.add_module('pre_relu1', nn.ReLU())
        self.specific1.add_module('pre_drop1', nn.Dropout(d))

        self.specific2 = nn.Sequential()
        self.specific2.add_module('fc2', nn.Linear(l1, l2))#specific feature of source domain 2
        self.specific2.add_module('pre_relu2', nn.ReLU())
        self.specific2.add_module('pre_drop2', nn.Dropout(d))

        self.predictor1 = nn.Sequential()
        self.predictor1.add_module('pre_fc1', nn.Linear(l2, 1))#predictor for source domain 1
        self.predictor1.add_module('pre_sigm1', nn.Sigmoid())

        self.predictor2 = nn.Sequential()
        self.predictor2.add_module('pre_fc2', nn.Linear(l2, 1))#predictor for source domain 2
        self.predictor2.add_module('pre_sigm2', nn.Sigmoid())


        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(l1, l2))
        self.domain_classifier.add_module('d_relu1', nn.ReLU())
        self.domain_classifier.add_module('d_fc3', nn.Linear(l2, 3))

    def forward(self,x1,x2,xt):

        f10,_=self.bgru1(x1)
        f11=self.cnn(x1)
        a=f11.shape[0]
        f11=f11.view(a,1,-1)
        f1=torch.cat([f10,f11],-1)#common features of source domain 1

        f20,_=self.bgru1(x2)
        f21=self.cnn(x2)
        f21=f21.view(a,1,-1)
        f2=torch.cat([f20,f21],-1)#common features of source domain 2

        ft0,_=self.bgru1(xt)
        ft1=self.cnn(xt)
        ft1=ft1.view(a,1,-1)
        ft=torch.cat([ft0,ft1],-1)#common features of target domain

        f1_s=self.specific1(f1)#specific features of source domain 1
        f2_s=self.specific2(f2)#specific features of source domain 2

        pre_f1=self.predictor1(f1_s)#HI prediction for source domain 1
        pre_f2=self.predictor2(f2_s)#HI prediction for source domain 2

        ft1=self.specific1(ft)
        ft2=self.specific1(ft)
        pre_t1=self.predictor1(ft1)
        pre_t2=self.predictor2(ft2)
        pre_t=(pre_t1+pre_t2)/2#ensemble HI prediction for target domain

        domain_output1 = []
        for i in range(int(x1.shape[0])):
            x0=f1[i,:,:].reshape(-1,l1)
            reverse_feature = ReverseLayerF.apply(x0, 1)
            domain_output1.append(self.domain_classifier(reverse_feature))
        domain_output2 = []
        for i in range(int(x2.shape[0])):
            x0=f2[i,:,:].reshape(-1,l1)
            reverse_feature = ReverseLayerF.apply(x0, 1)
            domain_output2.append(self.domain_classifier(reverse_feature))
        domain_output3 = []
        for i in range(int(xt.shape[0])):
            x0=ft[i,:,:].reshape(-1,l1)
            reverse_feature = ReverseLayerF.apply(x0, 1)
            domain_output3.append(self.domain_classifier(reverse_feature))
        return pre_f1, pre_f2, pre_t, f1, f2, ft, torch.stack(domain_output1,dim=1), \
               torch.stack(domain_output2,dim=1), torch.stack(domain_output3,dim=1)
