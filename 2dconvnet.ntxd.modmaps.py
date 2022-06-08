#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import torch
import mdtraj as md
import torch.nn as nn
import torch.nn.functional as F
import deeptime
from torch.utils.data import DataLoader
from tqdm import tqdm
from deeptime.decomposition.deep import VAMPNet
from deeptime.decomposition import VAMP
from copy import deepcopy

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
#torch.set_num_threads(12)

print(device)
print(torch.cuda.device_count())
##adjustable parameters
lagtime = 30
batch_size = 32
##

t = np.load("/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/VAMP/maps.npy")
t = t.astype(np.float32);t = t[::2]
t = torch.from_numpy(t)
t.to("cpu")
dataset = deeptime.util.data.TrajectoryDataset(lagtime=lagtime,trajectory=t)
n_val = int(len(dataset)*.3)
train_data,val_data = torch.utils.data.random_split(dataset,[len(dataset)-n_val,n_val])
loader_train = DataLoader(train_data, batch_size=batch_size,pin_memory=True,num_workers=16,shuffle=True)
loader_val = DataLoader(val_data,batch_size=len(val_data),pin_memory=True,num_workers=16,shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,2,padding=2)
        self.conv2 = nn.Conv2d(32,64,2,padding=2)
        self.conv3 = nn.Conv2d(64,128,2,padding=2)
        self.to_linear=None
        x = torch.randn(50,21).view(-1,1,50,21)
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear,100)
        self.fc2 = nn.Linear(100,7)
    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        if self.to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x.view(-1,self._to_linear)
    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim=1)
    
n = Net()
lobe = nn.DataParallel(n,device_ids = [0,1,2,3])
#lobe_timelagged = deepcopy(lobe).to(device)
lobe = lobe.to(device)
vampnet = VAMPNet(lobe=lobe,lobe_timelagged=None, learning_rate=5e-6, device=device)
m = vampnet.fit(loader_train, n_epochs=200,
                    validation_loader=loader_val, progress=tqdm).fetch_model()

np.save("/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/VAMP/current/trainscores2", vampnet.train_scores)
np.save("/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/VAMP/current/validationscores2", vampnet.validation_scores)
torch.save(vampnet.optimizer.state_dict(), "/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/VAMP/current/optim_statedict2.pt")
torch.save(lobe.state_dict(), "/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/VAMP/current/nonlagged_statedict2.pt")
#torch.save(lobe_timelagged.state_dict(), "/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/VAMP/current/timelagged_statedict2.pt")
torch.save(m,"/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/VAMP/current/model.pt")
#model = deepcopy(m)
#prob = model.transform(t)
#np.save("/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/VAMP/prob.npy",prob)
#v=VAMP(lagtime=lagtime,observable_transform = model).fit_fetch(dataset)
#proj = v.transform(t.numpy())
#np.save("/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/VAMP/proj.npy",proj)

