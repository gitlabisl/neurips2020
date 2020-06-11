


import torch
import torch.nn as nn
import numpy as np
import scipy
from scipy.io import loadmat

# load data here and normalize
X_subj = (X_subj - X_subj.max())/(X_subj.max() - X_subj.min())

#  Second order HOR model
class model(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,batch_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.w_i = nn.Linear(self.input_dim,self.hidden_dim)
        self.w_1 = nn.Linear(self.hidden_dim+self.input_dim,self.hidden_dim)
        self.w_2 = nn.Linear(self.hidden_dim+self.input_dim,self.hidden_dim)
        self.w_o = nn.Linear(self.hidden_dim,self.output_dim)
        self.h_1 = torch.zeros(self.batch_size,self.hidden_dim)
        self.h_2 = torch.zeros(self.batch_size,self.hidden_dim)


    def forward(self,x):
        timesteps,batch,_ = x.size()
        self.out = []
        for i in range(timesteps):
            self.G_1 = torch.tanh(self.w_1(torch.cat((x[i],torch.tanh(self.h_1)),1 )))
            self.G_2 = torch.tanh(self.w_2(torch.cat((x[i],torch.tanh(self.h_2)),1 )))
            self.h = torch.sigmoid(self.G_1*self.h_1 + self.G_2*self.h_2 + self.w_i(x[i]))
            self.h_1 = self.h
            self.h_2  = self.h_1
            self.out.append(torch.tanh(self.w_o(self.h)))
        return torch.cat(self.out).view(timesteps,batch,-1),self.h

for s in range(X_subj.shape[0]):
    for channel in range(5):
        print('For CHannel : ',channel)
        X_k = X_subj[s,:,:,channel,:]
        loss_fn = nn.MSELoss()

        # input and output shape:[sequence,batch,feature]
        hidden_size = 50

        data = torch.Tensor(X_k)
        batch_size = 12
        num_batches = int(X_k.shape[1]/batch_size)
        net = model(1,hidden_size,1,batch_size)
        optimizer = torch.optim.SGD(net.parameters(),lr = 0.001)

        # Training the network
        for i in range(7):
            Loss=[]
            Hidden = []
            for j in range(num_batches):	
                y_out,hidden = net(data[:,j*batch_size:(j+1)*batch_size,:])    # see if it is not a multiple
                Hidden.extend(hidden.detach().numpy())

                loss = loss_fn(y_out[:-1,:,:],data[1:,j*batch_size:(j+1)*batch_size,:])
                Loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                optimizer.step()
                print("Loss = ",sum(Loss)/len(Loss))
        Hidden = np.array(Hidden)
        Hidden = np.expand_dims(Hidden,axis = 2)
        if channel ==0 :
            Hidden_final = Hidden
        else:
            Hidden_final = np.concatenate((Hidden_final,Hidden),axis=2)
    Hidden_final = np.expand_dims(Hidden_final,axis=0)
    if s ==0 :
        Hidden_final_subj = Hidden_final
    else:
        Hidden_final_subj   = np.concatenate((Hidden_final_subj,Hidden_final),axis=0)
    Hidden_final=[]
    print(Hidden_final_subj.shape)


