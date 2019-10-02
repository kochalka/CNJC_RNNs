import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
from torch import optim

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

class RateUnitNetwork(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim, 
                 output_dim,
                 dt = 0.1, 
                 tau = 1, 
                 g = 1.5, 
                 prob_conn = 0.25, 
                 include_bias = False, 
                 noise_amp = None, 
                 device = torch.device('cpu')
                ):        
        super(RateUnitNetwork, self).__init__()
        # Cram everything into self
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim        
        self.output_dim = output_dim
        self.noise_amp = noise_amp
        self.tau = tau
        self.tanh = nn.Tanh()
        self.dt = dt
        self.device = device
        
        # Initialize layers
        self.i2h = nn.Linear(input_dim, hidden_dim, bias=include_bias)
        self.h2h = nn.Linear(hidden_dim, hidden_dim, bias=include_bias)
        self.h2o = nn.Linear(hidden_dim, output_dim, bias=include_bias)

        #TODO: torch.nn.init
        # Initialize weights
        # Recurrent connections
        size_h2h = (hidden_dim, hidden_dim)
        indices = np.where(np.random.uniform(0,1,size_h2h) < prob_conn)
        indices = np.delete(indices, np.where(indices[0]==indices[1]), axis=1) # remove self-connections
        mask = np.zeros(size_h2h, 'float')
        mask[indices[0],indices[1]] = 1
        self.h2h_mask = Variable(torch.from_numpy(mask).float()).to(device=device)
        indices = torch.from_numpy(indices).long()
        indices = indices.contiguous()
        sigma = g / np.sqrt(prob_conn * hidden_dim)
        values = torch.FloatTensor(int(indices.view(-1).size(0)/2)).normal_(std=sigma)
        self.h2h.weight.data = torch.sparse.FloatTensor(indices, 
                                                        values, 
                                                        torch.Size([hidden_dim, hidden_dim])
                                                       ).to_dense()
        self.h2h.weight.register_hook(self._h2h_backward_hook)
        # input and output layers
        i2h = (1 / np.sqrt(input_dim)) * np.random.randn(hidden_dim, input_dim)
        self.i2h.weight.data = torch.from_numpy(i2h).float()
        h2o = (1 / np.sqrt(hidden_dim)) * np.random.randn(output_dim, hidden_dim)
        self.h2o.weight.data = torch.from_numpy(h2o).float()

        # initialize hidden unit activity and output to None
        self.hiddens = None
        self.outputs = None
        
    def _h2h_backward_hook(self, grad):
        return grad * self.h2h_mask
        
    def step(self, input, hidden):        
        if self.noise_amp is not None:#NOT TESTED
            noise = torch.from_numpy(
                self.noise_amp*np.random.randn(self.hidden_dim)*np.sqrt(self.dt)
            ).float().to(self.device)
        else:
            noise = 0.
        
        #noise = noise_amp*randn(numUnits,1)*sqrt(dt);
		#Xv_current = WXX*X + WInputX*Input + noise;
		#Xv = Xv + ((-Xv + Xv_current)./tau)*dt;	
		#X = sigmoid(Xv);
		#Out = WXOut*X;
        
        # Should be fixed now -- 
        # Hidden is voltage. Then we take tanh to get rate where we need it.
        hidden = (1-self.dt/self.tau)*hidden +\
                 (self.dt/self.tau)*(self.h2h(self.tanh(hidden)) + self.i2h(input) + noise)
        

        output = self.h2o(self.tanh(hidden))
        return output, hidden    

    def forward(self, input, hidden):
        B, T, D = input.shape
        # Try to speed up by not reallocating a variable on every forward pass?
        if (self.hiddens is None) or not (self.hiddens.shape[:2] == (B,T)):
            self.hiddens = Variable(torch.zeros(B, T, self.hidden_dim, device=self.device))
            self.outputs = Variable(torch.zeros(B, T, self.output_dim, device=self.device))
        for t in range(T):
            outputs, hidden = self.step(input[:,t,:], hidden)
            self.outputs[:,t,:] = outputs
            self.hiddens[:,t,:] = hidden
        return self.outputs, self.hiddens

def save_checkpoint(state, i, base_dir, base_name_pfx):
    file_name = os.path.join(base_dir, base_name_pfx + ('_%i.pt' % i))
    torch.save(state, file_name)