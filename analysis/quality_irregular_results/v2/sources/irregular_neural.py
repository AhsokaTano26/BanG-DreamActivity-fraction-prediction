"""Observed-time GRU and a piecewise-linear neural CDE with explicit Heun solver."""
import torch
from torch import nn


def heun_step(field,state,increment,substeps=1):
    if substeps<1:raise ValueError('substeps must be positive')
    dx=increment/substeps
    for _ in range(substeps):
        k1=torch.bmm(field(state),dx.unsqueeze(-1)).squeeze(-1)
        k2=torch.bmm(field(state+k1),dx.unsqueeze(-1)).squeeze(-1)
        state=state+(k1+k2)/2
    return state


class IrregularForecaster(nn.Module):
    def __init__(self,architecture,features,width=32):
        super().__init__()
        if architecture not in ['time_gru','neural_cde']:raise ValueError(architecture)
        self.architecture=architecture;self.width=width;self.substeps=1
        self.initial=nn.Sequential(nn.Linear(4,width),nn.Tanh())
        if architecture=='time_gru':self.gru=nn.GRUCell(8,width)
        else:self.field_net=nn.Sequential(nn.Linear(width,width),nn.Tanh(),nn.Linear(width,width*4),nn.Tanh())
        self.tab=nn.Sequential(nn.Linear(features,width),nn.GELU(),nn.Dropout(.2))
        self.rank=nn.Embedding(2,4)
        self.head=nn.Sequential(nn.Linear(width*5+4,width),nn.GELU(),nn.Dropout(.2),nn.Linear(width,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def field(self,z):return self.field_net(z).reshape(-1,self.width,4)

    def encode(self,path,mask):
        batch,slots,length,channels=path.shape
        x=path.reshape(batch*slots,length,channels);valid=mask.reshape(batch*slots,length)
        # Sanitize before arithmetic: arbitrary padding must never affect state.
        x=torch.where(valid.unsqueeze(-1),x,torch.zeros_like(x))
        z=self.initial(x[:,0])*valid[:,0,None]
        for i in range(1,length):
            active=valid[:,i]&valid[:,i-1]
            delta=torch.where(active[:,None],x[:,i]-x[:,i-1],torch.zeros_like(x[:,i]))
            if self.architecture=='time_gru':new=self.gru(torch.cat([x[:,i],delta],-1),z)
            else:new=heun_step(self.field,z,delta,self.substeps)
            z=torch.where(active[:,None],new,z)
        return z.reshape(batch,slots,self.width)

    def forward(self,path,mask,tab,rank):
        z=self.encode(path,mask);batch=torch.arange(len(z),device=z.device)
        own=z[batch,rank];peer=z[batch,1-rank]
        return self.head(torch.cat([own,peer,own-peer,own*peer,self.tab(tab),self.rank(rank)],-1)).squeeze(-1)
