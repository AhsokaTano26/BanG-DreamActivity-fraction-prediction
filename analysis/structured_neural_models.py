"""Task-specific paired encoders; predict log remaining-average-rate multiplier.

Patching and availability/age inputs are inspired by time-series literature;
these are task adaptations, not reproductions of PatchTST or GRU-D.
"""
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURES=['shared_mlp','rank_mlp','dual_gru','dual_tcn','multiscale_tcn','cross_patch','gated_tcn']


class ConvBlock(nn.Module):
    def __init__(self,inputs,width,dilation):
        super().__init__();self.a=nn.Conv1d(inputs,width,3,padding=2*dilation,dilation=dilation)
        self.b=nn.Conv1d(width,width,3,padding=2*dilation,dilation=dilation)
        self.skip=nn.Conv1d(inputs,width,1) if inputs!=width else nn.Identity();self.dropout=nn.Dropout(.2)

    def forward(self,x):
        n=x.shape[-1];y=self.dropout(F.gelu(self.a(x)[...,:n]));y=self.dropout(F.gelu(self.b(y)[...,:n]))
        return y+self.skip(x)


class StructuredForecaster(nn.Module):
    def __init__(self,architecture,tab_features,width=48):
        super().__init__()
        if architecture not in ARCHITECTURES:raise ValueError(architecture)
        self.architecture=architecture;self.width=width
        self.tab=nn.Sequential(nn.Linear(tab_features,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(.2),nn.Linear(width,width),nn.GELU())
        self.rank_embedding=nn.Embedding(2,8)
        dynamic=0
        if architecture=='dual_gru':
            self.encoder=nn.GRU(8,width,num_layers=2,dropout=.2,batch_first=True);dynamic=width*4
        elif architecture in ['dual_tcn','multiscale_tcn','gated_tcn']:
            self.encoder=nn.Sequential(*[ConvBlock(8 if i==0 else width,width,2**i) for i in range(3)])
            dynamic=width*(8 if architecture=='multiscale_tcn' else 4)
        elif architecture=='cross_patch':
            self.patch=nn.Linear(8*8,width);self.position=nn.Parameter(torch.randn(1,1,8,width)*.01)
            self.slot=nn.Embedding(2,width);self.query=nn.Embedding(2,width)
            layer=nn.TransformerEncoderLayer(width,4,2*width,dropout=.2,batch_first=True,norm_first=True,activation='gelu')
            self.attention=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);dynamic=width
        if architecture in ['dual_gru','dual_tcn','multiscale_tcn','gated_tcn']:
            self.summary=nn.Linear(2*width,width)
        inputs=width+8+dynamic
        self.fusion=nn.Sequential(nn.Linear(inputs,width*2),nn.GELU(),nn.Dropout(.2))
        if architecture=='gated_tcn':
            self.gate=nn.Linear(width+8,3)
            self.experts=nn.ModuleList([nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,2)) for _ in range(3)])
            for expert in self.experts:nn.init.zeros_(expert[-1].weight);nn.init.zeros_(expert[-1].bias)
        else:
            self.head=nn.Linear(width*2,1 if architecture=='shared_mlp' else 2)
            nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def encode(self,pair,rank):
        n,_,length,_=pair.shape;x=pair.reshape(n*2,length,8);valid=x[:,:,2]>0
        if self.architecture=='dual_gru':encoded,_=self.encoder(x)
        else:encoded=self.encoder(x.transpose(1,2)).transpose(1,2)
        mean=(encoded*valid.unsqueeze(-1)).sum(1)/valid.sum(1).clamp_min(1).unsqueeze(-1)
        indices=torch.arange(length,device=x.device).unsqueeze(0).expand(len(x),-1)
        last=(indices*valid).amax(1)
        final=encoded[torch.arange(len(x),device=x.device),last]
        summary=self.summary(torch.cat([mean,final],-1))*valid.any(1).unsqueeze(-1)
        summary=summary.reshape(n,2,self.width);index=torch.arange(n,device=x.device)
        own,peer=summary[index,rank],summary[index,1-rank]
        return torch.cat([own,peer,own-peer,own*peer],-1)

    def forward(self,pair,recent,tab,rank):
        static=self.tab(tab);embedding=self.rank_embedding(rank);parts=[static,embedding]
        kind=self.architecture
        if kind in ['dual_gru','dual_tcn','multiscale_tcn','gated_tcn']:
            parts.append(self.encode(pair,rank))
            if kind=='multiscale_tcn':parts.append(self.encode(recent,rank))
        elif kind=='cross_patch':
            n=len(pair);patches=pair.reshape(n,2,8,8*8)
            slots=self.slot(torch.arange(2,device=pair.device)).reshape(1,2,1,self.width)
            tokens=(self.patch(patches)+self.position+slots).reshape(n,16,self.width)
            available=(pair[:,:,:,2].reshape(n,2,8,8)>0).any(-1).reshape(n,16)
            query=self.query(rank).unsqueeze(1)
            padding=torch.cat([torch.zeros(n,1,dtype=torch.bool,device=pair.device),~available],1)
            parts.append(self.attention(torch.cat([query,tokens],1),src_key_padding_mask=padding)[:,0])
        fused=self.fusion(torch.cat(parts,-1))
        if kind=='gated_tcn':
            gate=self.gate(torch.cat([static,embedding],-1)).softmax(-1)
            outputs=torch.stack([expert(fused) for expert in self.experts],1)
            heads=(outputs*gate.unsqueeze(-1)).sum(1)
        else:heads=self.head(fused)
        return heads[:,0] if kind=='shared_mlp' else heads.gather(1,rank[:,None]).squeeze(1)
