# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:11:31 2021

@author: nnak
"""

import torch 
import math

def GP_regularization(_v,_bin_boundaries,GP_sigma,GP_lambda,cholesky=True):
    '''
    INPUT: 
        _v: Tensor-> velocities (TxNxD), 
        _bin_boundaries (T+1): Vector-> time bin boundaries, 
        GP_sigma (D): Vector-> RBF kernel noise standard deviation per latent dimension D, 
        GP_lambda (D): Vector-> sqrt of lengthscale of the RBF kernel per latent dimension D, 
        cholesky: flag controling how the inverse is to be computed
    
    RETURNS:                ______________
        log_prior: Scalar-> LOG LIKELIHOOD of Normal distribution Prior with covariance based on a GP with RBF Kernel
                            ##############
    '''
    
    input_size=_v.shape[1]
    pi=torch.tensor(math.pi)

    # Create the time differences TxT matrix
    middle_bounds=(_bin_boundaries[1:]+_bin_boundaries[0:-1])/2
    
    # Calculte inverse of variance for the RBF
    inv_GP_var=1/(2*(GP_sigma*GP_sigma))
    
    # Calculate Kernel/Covariance Matrix
    T_matrix=((GP_lambda*GP_lambda)*torch.exp(-inv_GP_var*((middle_bounds.unsqueeze(1)-middle_bounds)**2).unsqueeze(2))).transpose(2,0)
    
    # Calcualte Inverse
    if cholesky:
        inv_T_matrix=torch.cholesky_inverse(torch.linalg.cholesky(T_matrix))
    
    else:
        inv_T_matrix=torch.linalg.inv(T_matrix)
        
    # Reshape Velocities for Broadcasting
    # TxNxD -> NxDxT
    _v=_v.permute(1,2,0)

    # (Velocity x SIGMA^-1 x Velocity^T)
    # NxDx1xT with 1xDxTxT -> NxDx1xT -> sum and squezee -> NxD
    log_exp=((_v.unsqueeze(2)@inv_T_matrix.unsqueeze(0))*(_v.unsqueeze(2))).sum(-1).squeeze(-1)
    log_det=input_size*(torch.log(torch.linalg.det(T_matrix))+T_matrix.shape[0]*torch.log(2*pi))
    log_prior=-0.5*(log_exp.sum()+log_det.sum())
    
    
    return log_prior
 
   
dim=2
N=5
T=10
log_prior=GP_regularization(torch.nn.Parameter(torch.rand(T,N,dim)),torch.arange(T+1),torch.nn.Parameter(torch.ones(dim)),torch.nn.Parameter(torch.ones(dim)),cholesky=True)
print(log_prior)


    
