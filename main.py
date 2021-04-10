import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import scipy.io as scio
from Model import msdan
import MMD
import time
from data_process import *
# begin = time.clock()

batch_size = 20
LR=0.0001#leanring rate
#loading data
data_scr1, label_scr1=get_dataset(r'./data/c2.mat','c2')#source doamin 1
data_scr2, label_scr2=get_dataset(r'./data/c3.mat','c3')#source doamin 2
data_tar_tr,label_tar_tr=get_dataset(r'./data/c1_tr1.mat','c1_tr')#target domain for training
test_data,test_label = get_dataset(r'./data/c1_te1.mat', 'c1_te')#target domain for testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#parameter setup
hidden_size=160
layer_size=2
input_size = 360
in_channel=1
out_channel=16
kernel=3

#mdsan
model=msdan(input_size, hidden_size,layer_size,in_channel,out_channel, kernel)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters
loss_func0 = nn.MSELoss(reduce=True, size_average=True)
loss_func1=nn.CrossEntropyLoss()

predict_ = []
real_ = []
Pre_loss = []
mmd=[]
ad_loss=[]
# #
for n in range(500):#training
    d_scr1,l_scr1=data_generate(data_scr1, label_scr1)
    d_scr2,l_scr2=data_generate(data_scr2, label_scr2)
    d_tar_tr,l_tar_tr= data_generate(data_tar_tr,label_tar_tr)
    pref1,pref2,pret,f1,f2, ft, domain_pre1,domain_pre2,domain_pret,=model(d_scr1,d_scr2,d_tar_tr)
    #prediction
    pre_loss_s1 = loss_func0(pref1, l_scr1)
    pre_loss_s2 = loss_func0(pref2, l_scr2)
    pre_loss_t=loss_func0(pret, l_tar_tr)
    pre_loss=pre_loss_t+pre_loss_s1+pre_loss_s2

    # adversarial learning
    shape1=d_scr1.shape
    domain_label_s1 = torch.zeros(int(shape1[0]*shape1[1])).long().cuda()
    domain_label_s2 = (1*torch.ones(int(shape1[0]*shape1[1]))).long().cuda()
    domain_label_t = (2*torch.ones(int(shape1[0]*shape1[1]))).long().cuda()
    domain_pre1 = torch.reshape(domain_pre1,(batch_size*TIME_STEP,3))
    d_loss_s1 = loss_func1(domain_pre1,domain_label_s1)
    domain_pre2 = torch.reshape(domain_pre2,(batch_size*TIME_STEP,3))
    d_loss_s2 = loss_func1(domain_pre2,domain_label_s2)
    domain_t = torch.reshape(domain_pret,(batch_size*TIME_STEP,3))
    d_loss_t = loss_func1(domain_t,domain_label_t)
    d_loss=d_loss_s1+d_loss_s2+d_loss_t

    # #training with mmd
    loss_MMD1=0
    loss_MMD2=0
    F1=f1.reshape([-1,640])
    F2=f2.reshape([-1,640])
    Ft=ft.reshape([-1,640])
    loss_mmd1 = MMD.mix_rbf_mmd2(F1, Ft,[1000])#MMD loss between the source domain 1 and the target domain
    loss_mmd2 = MMD.mix_rbf_mmd2(F2, Ft,[1000])#MMD loss between the source domain 2 and the target domain
    loss_MMD1 += loss_mmd1
    loss_MMD2 += loss_mmd2
    loss_MMD=loss_MMD1+loss_MMD2 #total MMD loss

    #total training loss
    Loss=pre_loss+0.2*loss_MMD+0.02*d_loss
    optimizer.zero_grad()  # clear gradients for this training step
    Loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    Loss = Loss.cuda().data.cpu().detach().numpy()
    loss_pre=pre_loss.cuda().data.cpu().numpy()
    loss_d = d_loss.cuda().data.cpu().numpy()
    loss_MMD=loss_MMD.cuda().data.cpu().numpy()

    if n % 50 == 0:
        print("step {}| training total loss:{},"
              "pre_loss:{}, "
              "class_loss:{}, "
              "loss on feature MMD:{}, ".format(n, Loss,loss_pre,loss_d,loss_MMD))
    Pre_loss.append(loss_pre)
    mmd.append(loss_MMD)
    ad_loss.append(loss_d) #save the training loss

np.save(r'./result/pre_loss1_1.npy', Pre_loss)
np.save(r'./result/mmd1_1.npy', mmd)
np.save(r'./result/ad_loss1_1.npy',  ad_loss)

path = r'./model/model1_1'
torch.save(model.state_dict(), path)
print('Training finished')
# end = time.clock()
# print('training time',end - begin)

# begin1 = time.clock()
path=r'./model\model1_1'
model.load_state_dict(torch.load(path))
model.eval()
x_np = test_data
y_np = test_label
x = torch.from_numpy(x_np).cuda()
x=x.unsqueeze(dim=1)
y = torch.from_numpy(y_np).cuda()
y = y.cuda().data.cpu().numpy()
_,_,pre ,_ ,_,_,_,_ ,_   =model(x,x,x)
pre=pre.cuda().data.cpu().detach().numpy()
pre_data=pre.squeeze()
print(pre_data.shape)
np.save(r'.\result/prediction1_1.npy',  pre_data)
np.save(r'.\result/real_value1_1.npy', y)

# torch.cuda.empty_cache()
# end1 = time.clock()
# print('testing time',end1 - begin1)











