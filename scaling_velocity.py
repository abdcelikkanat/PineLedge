# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:29:34 2022

@author: nnak
"""

import torch 
import numpy as np
from torch_sparse import spspmm

# REAL VERSION vectorizing
#################################################################
#################################################################

L=1/3
max_b=3
D=2
# multi indexing over list
# L = list(range(0, 101, 10))  # works in Python 2 or 3

# from operator import itemgetter
# itemgetter(2, 5)(L)


# nodes = 3
N=45

# events of an N=3 system, defining the 3 upper triu pairs
events=torch.load('C:/Users/nnak/Documents/noname/noname/src/list_events') # events is a list of list, [[0.2, 0.2], [0.3, 0.4]]
pairs=torch.load('C:/Users/nnak/Documents/noname/noname/src/node_pairs').transpose(0,1) # [[1,2], [0,2]]


def link_l(events,pairs):
    
    # calculate lengths of interaction of each individual node pair in the dataset -> Tensor with size Pairs_interacting x 1
    lengths=torch.tensor([len(l) for l in events])
    
    # concatentate events lists -> Tensor of size #Events
    events_t=torch.from_numpy(np.concatenate(events)) # [0.2, 0.3, 0.4, 0.1]
    
    
    
    # create i,j,t pairs for all events and dyads
    # repeat interleave repeats pairs wrt to the number of times they interacted
    i_j=torch.repeat_interleave(pairs, lengths, dim=0) # ---> Check expand function
    
    # div operation between interaction times and window lenghth
    # returns how many intervals have passed until the interaction -> Tensor #E x 1
    intervals_passed=torch.div(events_t,L,rounding_mode='floor').unsqueeze(1)
    
    # modulo operation between event time and bin size to get time left in the current interval-> Tensor #E x 1
    residual_time=(events_t%L).unsqueeze(1)
    
    # dyad + intervals that have passed creating a tria -> Tensor #E x 3
    i_j_b=torch.cat((i_j,intervals_passed),1).long()
    
    
    
    # velocities, simply the torch parameters
    v=torch.randn(N,D,max_b)
    
    # torch parameters for initial position
    x0=torch.randn(N,D)
    
    
    # distance covered per time bin aggregated
    
    x_t=(v*L).cumsum(-1)+x0.unsqueeze(-1)
    
    # events calculation
    ##        ---------------Dx wrt to all bins passed----------------| |-------------Dv*time left------------------------------------------|
    ll_events=x_t[i_j_b[:,0],:,i_j_b[:,2]]-x_t[i_j_b[:,1],:,i_j_b[:,2]]+(v[i_j_b[:,0],:,i_j_b[:,2]]-v[i_j_b[:,1],:,i_j_b[:,2]])*residual_time
    
    return ll_events
    
    

# Batch Sampling

#Problems when we have event on t=0 it gets absorved by the sparsity
########################################

# size of the sparse matrix
Rows=N**2
lengths=torch.tensor([len(l) for l in events])
events_t=torch.from_numpy(np.concatenate(events))

i_j=torch.repeat_interleave(pairs, lengths, dim=0)


sparse_row=(i_j[:,0]*N)+i_j[:,1]

sampling_weights=torch.ones(N)
sample_size=int(0.1*N)
sample_idx=torch.multinomial(sampling_weights,sample_size,replacement=False)


#sample_triu=torch.triu_indices(sample_size,sample_size,1)
pairs=((sample_idx*N).unsqueeze(1)+sample_idx).reshape(-1).unsqueeze(0)#[sample_triu.unbind()].unsqueeze(0)


# random sample location matrix and mask

sample_pair_mask = torch.sparse_coo_tensor(pairs.repeat(2,1), torch.ones(pairs.shape[1]), (Rows, Rows))


# total edges matrix in sparse N^2 x Projection

sparse_edges=torch.sparse_coo_tensor(torch.cat((sparse_row.unsqueeze(0),torch.arange(events_t.shape[0]).unsqueeze(0)),0).long(), events_t, (Rows, events_t.shape[0]))

## ADD constant c to the events so t=0 event survive and then substract it from valueC
c=1
indexC, valueC=spspmm(pairs.repeat(2,1),torch.ones(pairs.shape[1]),torch.cat((sparse_row.unsqueeze(0),torch.arange(events_t.shape[0]).unsqueeze(0)),0).long(),(events_t+c),Rows,Rows,events_t.shape[0],coalesced=True)
valueC=valueC-c

sample_i=torch.div(indexC[0],N,rounding_mode='floor').unsqueeze(1)
sample_j=(indexC[0]%N).unsqueeze(1)


intervals_passed_sample=torch.div(valueC,L,rounding_mode='floor').unsqueeze(1)

# time left in the current interval
residual_time_sample=(valueC%L).unsqueeze(1)

sample_i_j_b=torch.cat((sample_i,sample_j,intervals_passed_sample),1).long()

print(sample_i_j_b.shape)
print(i_j_b.shape)



