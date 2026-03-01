import torch
import numpy as np
from tqdm import tqdm
from torch import einsum
from torch.linalg import solve, inv
from scipy.sparse.linalg import svds
from BFM.md import mirror_descent

def Initialization(X, r):
    
    n, P = X.shape
    
    k = min(n, P)
    
    U, S, _  = svds(X.T / (n-1) ** 0.5, k - 1)
    
    if S[0] < S[1]:
    
        U = U[:, ::-1]
        
        S = S[::-1]
        
    sigma2_estimator = (S[r : ] ** 2).sum() / (len(S) - r)
    
    mu = U[:,0:r] * np.sqrt(S[0:r] ** 2 - sigma2_estimator)
    
    return mu, float(sigma2_estimator)

def sigma_update(X, mu, Cov, v, mu_eta, Psi, L, a_sigma, b_sigma):
    
    _, n = mu_eta.size()
    
    np_sigma = b_sigma + 0.5 * (X.T - mu @ mu_eta).square().sum(1) + \
        0.5 * n * (mu * (mu @ Psi)).sum(1) + \
            0.5 * v / (v-2) * einsum('pij,jk-> p', Cov, L)
        
    return np_sigma, (a_sigma + 0.5 * n) / np_sigma


def eta_update(mu_eta, Psi, X, C, mu, Cov, v):
    
    P,r = mu.size()
        
    device = mu.device
    
    muTC = mu.T * C
    
    Psi = torch.eye(r, device = device, dtype = torch.float64) +  muTC @ mu +   (C.view(-1,1,1) * (v / (v - 2)).view(-1,1,1)  * Cov).sum(0)
    
    mu_eta = solve(Psi, muTC @ X.T)
            
    return mu_eta, Psi


def shrinkage(param, a, c1, c2):
    
    P,r = param.size()
    
    device = param.device
    
    weight = torch.linspace(1, r, steps = r, device = device, dtype = torch.float64)
    
    ink = param.abs().sqrt()
    
    return (P + 0.5 * a  + 0.5 * weight.pow(c1)) / (torch.maximum(param.abs().pow(1.5), torch.tensor(1e-5, device = device)) * (ink.sum(0) + 1 / weight.pow(c2)))
    

def B_update(X, mu, Precision, mu_eta, L, v, a, c1, c2, C):
    
    _,r = mu.size()
    n,_ = X.size()
    
    XTmu_eta = X.T @ mu_eta.T
    
    mu_old = torch.zeros_like(mu)
    
    for i in range(1,50):
        
        if (mu - mu_old).norm(p=float('inf')) < 1e-5:
            
            break
        
        else:
            
            mu_old = mu.clone()
            
            lr1 =  (1 / n) * (i + 30) ** (-0.75)
            lr2 =  (1 / n) * (i + 30) ** (-0.75)
        
            Lambda = shrinkage(mu, a, c1, c2)
        
            mu.add_(solve(Precision, C.view(-1,1) * (mu @ L - XTmu_eta)  + mu * Lambda), alpha = -lr1)
        
            Precision.mul_(1 - lr2).add_((v / (v - 2)).view(-1,1,1) * (C.view(-1,1,1) * L.view(1,r,r) + torch.diag_embed(Lambda)), alpha = lr2)
            
    return mu, Precision


def NGVI(X, device, r = 50, a = 10, c1 = 2.3, c2 = 0.7, score = False):
    
    if a <= 4:
        raise ValueError("a should larger than 4")
    
    if c1 + c2 <= 0.25:
        raise ValueError("c1 + c2 should larger than 0.25")
    
    a_sigma = 1
    b_sigma = 1
    
    n,P = X.shape
    
    ## Initialization
    mu, sigma2_estimator = Initialization(X, r)
    
    X = torch.from_numpy(X).to(device).to(torch.float64)
    
    mu = torch.from_numpy(mu).to(device).to(X.dtype)
    
    C = torch.ones(P, device = device, dtype = torch.float64) / sigma2_estimator
    Precision =  1e2 * torch.eye(r, device = device, dtype = torch.float64).repeat(P, 1, 1)
    mu_eta = torch.zeros(r, n, device = device, dtype = torch.float64)
    Psi =  torch.eye(r, device = device, dtype = torch.float64)
    v = 1000 * torch.ones(P,device = device, dtype = torch.float64)
    
    Cov = inv(Precision)
    
    for i in tqdm(range(60)):
        
        # Update eta 
        mu_eta, Psi = eta_update(mu_eta, Psi, X, C, mu, Cov, v)
        
        S = inv(Psi)
        
        L = mu_eta @ mu_eta.T + n * S
        
        # Update sigma2 
        np_sigma, C = sigma_update(X, mu, Cov, v, mu_eta, S, L, a_sigma, b_sigma)
        
        # Update B
        
        #Update mu and Precision
        mu, Precision = B_update(X, mu, Precision, mu_eta, L, v, a, c1, c2, C)
        
        Cov = inv(Precision)
        
        # Update v
        v = mirror_descent(v, C * einsum('pij,ji-> p', Cov, L), mu, torch.linalg.cholesky(Cov), a, c1, c2)
        
    
    if score == True:
        return mu.to('cpu'), Cov.to('cpu'), mu_eta.to('cpu'), np_sigma.to('cpu'), v.to('cpu')
    else:
        return mu.to('cpu'), Cov.to('cpu'), np_sigma.to('cpu'), v.to('cpu')    





        
        
        
        
        
        
        
        
        
        
        