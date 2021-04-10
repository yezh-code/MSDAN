import matplotlib.pyplot as plt
import numpy as np
font = {'family': 'Times New Roman', 'weight': 'normal', 'size':15}


##Plot the figure of the training loss
pre_loss=np.load(r'./result/pre_loss1_1.npy')
ad_loss=np.load(r'./result/ad_loss1_1.npy')
mmd=np.load(r'./result/mmd1_1.npy')
plt.figure()
plt.rc('font', family='Times New Roman')
fig1=plt.subplot(311)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(pre_loss)
fig1.set_xlabel('Epoch', font)
fig1.set_ylabel('Pre_loss', font)
fig2=plt.subplot(312)
plt.plot(ad_loss)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig2.set_xlabel('Epoch', font)
fig2.set_ylabel('Ad_loss', font)
fig3=plt.subplot(313)
plt.plot(mmd)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig3.set_xlabel('Epoch', font)
fig3.set_ylabel('MMD', font)
plt.savefig('./figure/loss.png')
plt.show()

#Plot the figure of prediction result
real_value=np.load(r'./result/real_value1_1.npy')
pre_value=np.load(r'./result/prediction1_1.npy')

fig=plt.figure(figsize=(5,3))
ax = fig.add_subplot(1, 1, 1)
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax.plot(real_value,color='k')
ax.plot(pre_value,color='r',marker='s')
ax.set_xlabel('Cycle Num.', font)
ax.set_ylabel('HI', font)
plt.savefig('./figure/prediction.png')
plt.show()

