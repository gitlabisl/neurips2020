import torch
import torch.nn as nn
import numpy as np
import scipy
from scipy.io import loadmat

# load data (31,250,24,5,1) where 31 subjects, 250 timesteps, 24 probes, 5 channels, 1 input size
X_subj = load_data()

# normalize
X_subj = (X_subj - X_subj.max())/(X_subj.max() - X_subj.min())

# classical lstm model
class model(nn.Module):
	def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
		super().__init__()
		self.input_size = input_sz
        self.hidden_size = hidden_sz
        # input gate
        self.W_ii = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hi = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = Parameter(torch.Tensor(hidden_sz))
        # forget gate
        self.W_if = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hf = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = Parameter(torch.Tensor(hidden_sz))
        # ???
        self.W_ig = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hg = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_g = Parameter(torch.Tensor(hidden_sz))
        # output gate
        self.W_io = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_ho = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = Parameter(torch.Tensor(hidden_sz))
         
        self.init_weights()
		
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states
        for t in range(seq_sz): # iterate over the time steps
            x_t = x[:, t, :]
            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


for s in range(X_subj.shape[0]):
	for channel in range(5):
		print('For CHannel : ',channel)
		X_k = X_subj[s,:,:,channel,:]
		loss_fn = nn.MSELoss()

		# input and output shape:[sequence,batch,feature]
		# data = torch.rand(9,20,5)
		hidden_size = 50

		data = torch.Tensor(X_k)
		batch_size = 12
		num_batches = int(X_k.shape[1]/batch_size)
		net = model(1,hidden_size,1,batch_size)
		optimizer = torch.optim.Adam(net.parameters(),lr = 0.001)
		# Training the network
		for i in range(4):
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


