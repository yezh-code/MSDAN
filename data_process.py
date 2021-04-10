import numpy as np
import scipy.io as scio
import torch
import random

TIME_STEP =1
batch_size = 20
def get_dataset(path,name):
    data = (scio.loadmat(path).get(name)).astype(np.float32)
    data = np.array(data)
    a = data[:, 1:]
    b = data[:, 0]
    return a,b
def data_generate(data,label):
    l=data.shape[0]
    batch_input = []
    batch_label = []
    for batch in range(batch_size):
        num = int(random.randint(1, int(l)-TIME_STEP))
        start, end = num, num + TIME_STEP  # time range
        x_np = data[start:end, :]
        y_np = label[start:end]
        batch_input.append(x_np)
        y_np = np.array(y_np)
        batch_label.append(y_np)
    batch_input = np.array(batch_input)
    x = torch.from_numpy(batch_input).cuda()  # shape (batch, time_step, input_size)
    y = batch_label
    y = np.array(y)
    y = y[:, :, np.newaxis]
    y = torch.from_numpy(y).cuda()
    return x,y



