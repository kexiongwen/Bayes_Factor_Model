import torch
from tqdm import tqdm
from torch.linalg import solve, inv
from BFM.PM_SVD import PM_SVD
from torch import einsum

def Initialization(X, n, r):
        
    U, S = PM_SVD(X.T / (n-1) ** 0.5, n)
    
    sigma2_estimator = S[r : ].square().sum() / (len(S) - r)
    
    mu = U[:,0:r] * (S[0:r].square() - sigma2_estimator).sqrt()
    
    return mu + 0.01 * torch.randn_like(mu), sigma2_estimator
    
    
def sigma_update(X, mu, Cov, v, mu_eta, Psi, L, a_sigma, b_sigma):
    
    _, n = mu_eta.size()
    
    np_sigma = b_sigma + 0.5 * (X.T - mu @ mu_eta).square().sum(1) + \
        0.5 * n * (mu * (mu @ Psi)).sum(1) + \
            0.5 * v / (v-2) * einsum('prr,rr-> p', Cov, L)
        
    return np_sigma, (a_sigma + 0.5 * n) / np_sigma
        
def eta_update(mu_eta, Psi, X, C, mu, Cov, v):
    
    P,r = mu.size()
        
    device = mu.device
    
    muTC = mu.T * C
    
    Psi = torch.eye(r, device = device, dtype = torch.float64) +  muTC @ mu + (C.view(P,1,1) *  Cov).sum(0) * (v / (v - 2))
    
    mu_eta = solve(Psi, muTC @ X.T)
            
    return mu_eta, Psi


def shrinkage(param, a, b, c):
    
    P,r = param.size()
    
    device = param.device
    
    weight = torch.linspace(1, r, steps = r, device = device, dtype = torch.float64).pow(0.25 + c)
    
    ink = param.abs().sqrt().mul_(weight)
    
    return ((P * r + 5 + 0.5 * a) * weight) / ((param.abs().pow(1.5) + 1e-6) * (ink.sum() + b))
    

def B_update(X, mu, Precision, mu_eta, L, v, a, b, c, C):
    
    _,r = mu.size()
    n,_ = X.size()
    
    XTmu_eta = X.T @ mu_eta.T
    
    mu_old = torch.zeros_like(mu)
    
    for i in range(1,200):
        
        if (mu - mu_old).norm(p=float('inf')) < 1e-5:
            
            break
        
        else:
            
            mu_old = mu.clone()
            
            lr1 =  (1 / n) * (i + 30) ** (-0.75)
            lr2 =  (1 / n) * (i + 30) ** (-0.75)
        
            Lambda = shrinkage(mu, a, b, c)
        
            mu.add_(solve(Precision, C.view(-1,1) * (mu @ L - XTmu_eta)  + mu * Lambda), alpha = -lr1)
        
            Precision.mul_(1 - lr2).add_(C.view(-1,1,1) * L.view(1,r,r) + torch.diag_embed(Lambda), alpha = lr2 * (v / (v - 2)))
            
    return mu, Precision


def NGVI(X, a = 1, b = 100, c = 0.25, v =  1000,  r = 50, a_sigma = 1, b_sigma = 1):
    
    n, P = X.size()
    
    X = X.to(torch.float64)
    
    device = X.device
    
    ## Initialization
    mu, sigma2_estimator = Initialization(X, n, r)
    C = torch.ones(P, device = device, dtype = torch.float64) / sigma2_estimator
    Precision =  1e2 * torch.eye(r, device = device, dtype = torch.float64).repeat(P, 1, 1)
    mu_eta = torch.zeros(r, n, device = device, dtype = torch.float64)
    Psi =  torch.eye(r, device = device, dtype = torch.float64)
    
    for i in tqdm(range(50)):
        
        Cov = inv(Precision)
        
        # Update eta 
        mu_eta, Psi = eta_update(mu_eta, Psi, X, C, mu, Cov, v)
        
        S = inv(Psi)
        
        L = mu_eta @ mu_eta.T + n * S
        
        # Update sigma2 
        np_sigma, C = sigma_update(X, mu, Cov, v, mu_eta, S, L, a_sigma, b_sigma)
        
        # Update B
        mu, Precision = B_update(X, mu, Precision, mu_eta, L, v, a, b, c, C)
        
    return mu, Precision, np_sigma  





        
        
        
        
        
        
        
        
        
        
        