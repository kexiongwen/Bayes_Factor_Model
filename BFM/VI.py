import torch
from tqdm import tqdm
from torch.linalg import solve, solve_triangular, cholesky, inv
from BFM.shrinkage import local_shrinkage
from torch.distributions.gamma import Gamma
from torch import einsum

def Sample_B(mu, R, v, S, extra_output = False):
    
    P,r = mu.size()
    
    device = mu.device
    
    u = Gamma(v/2 * torch.ones(P, 1, S, device = device, dtype = torch.float64), v/2).sample()
        
    B_sample = mu.view(P,r,1) + (solve_triangular(R, torch.randn(P, r, S, device = device, dtype = torch.float64), upper = True) / u.sqrt())
    
    if S == 1:
        B_sample = B_sample.squeeze(-1)
    
    if extra_output:
        return B_sample, u
    else:  
        return B_sample


def delta_and_sigma_update(X, mu, R, v, mu_eta, Psi, b, a_sigma, b_sigma):
    
    P,r = mu.size()
    
    _, n = mu_eta.size()
    
    device = mu.device
    
    np_sigma = torch.zeros(P, device = device, dtype = torch.float64)
    
    np_delta = 0
    
    weight = torch.linspace(1, r, steps = r, device = device, dtype = torch.float64).pow(0.3)
    
    for i in range(10):
        
        B_sample = Sample_B(mu, R, v, S = 100)
        
        np_sigma +=  (0.5 * ((X.T).view(P,n,1) - einsum('prs,rn-> pns',  B_sample , mu_eta)).square().sum(1) + 0.5 * n * (B_sample * einsum('prs,rr-> prs', B_sample, Psi)).sum(1) + b_sigma).sum(1) / 1000
        
        np_delta +=  ((B_sample.abs().sqrt() * weight.view(1,r,1)).sum((0,1)) + b).sum() / 1000
        
    return np_sigma, (a_sigma + 0.5 * n) / np_sigma, np_delta
        
                   
def eta_update(mu_eta, Psi, X, C, mu, R, v):
    
    P,r = mu.size()
        
    device = mu.device
    
    for i in range(1,100):
        
        #lr1 = (1 / P) * (1 / i)
        #lr2 = (1 / P) * (1 / i)
        
        lr1 =  (1 / P) * 0.01
        lr2 =  (1 / P) * 0.01 
         
        B_sample = Sample_B(mu, R, v, S = 1)
        
        mu_eta.add_(solve(Psi, B_sample.T @ (C.view(-1,1) * (B_sample @ mu_eta - X.T)) + mu_eta), alpha = -lr1)
        
        Psi.mul_(1 - lr2).add_((B_sample.T @ (C.view(-1,1) * B_sample) + torch.eye(r, device = device, dtype = torch.float64)), alpha = lr2)
        
    return mu_eta, Psi


def B_update(X, mu, Precision, mu_eta, Phi, np_delta, v, C):
    
    P,r = mu.size()
    n,_ = X.size()
    
    device = mu_eta.device
    
    for i in range(1,100):
        
        #lr1 = (1 / n) * (1 / i)
        #lr2 = (1 / n) * (1 / i)
        
        lr1 =  (1 / n) * 0.01
        lr2 =  (1 / n) * 0.01
        
        B_sample, u = Sample_B(mu, cholesky(Precision).transpose(1,2), v, S = 1,extra_output = True)
        
        lam = Gamma(2 * P * r + 5, np_delta).sample()
        
        Lambda = local_shrinkage(lam, B_sample).square()
        
        mu.add_(solve(Precision, C.view(-1,1) * ((B_sample @ mu_eta - X.T) @ mu_eta.T + n * B_sample @ Phi) + B_sample * Lambda) / u.view(-1,1), alpha = -lr1)
        
        Precision.mul_(1 - lr2).add_((C.view(-1,1,1) * (mu_eta @ mu_eta.T + n * Phi).view(1,r,r) + torch.diag_embed(Lambda)) / u, alpha = lr2)
        
    return mu, Precision


def NGVI(X, b, v =  1000, r = 50, a_sigma = 1, b_sigma = 1):
    
    n, P = X.size()
    
    X = X.to(torch.float64)
    
    device = X.device
    
    ## Initialization
    PCA = torch.pca_lowrank(X, q = r, center = True, niter= 10)
    mu = PCA[2] * PCA[1].sqrt()
    Precision =  1e2 * torch.eye(r, device = device, dtype = torch.float64).repeat(P, 1, 1)
    
    mu_eta = torch.zeros(r, n, device = device, dtype = torch.float64)
    Psi =  torch.eye(r, device = device, dtype = torch.float64)
    
    C = 10 * torch.ones(P, device = device, dtype = torch.float64)
    
    for i in tqdm(range(1000)):
        
        R = cholesky(Precision)
        
        # Update eta 
        mu_eta, Psi = eta_update(mu_eta, Psi, X, C, mu, R.transpose(1,2), v)
        
        S = inv(Psi)
        
        # Update delta and sigma2 
        np_sigma, C, np_delta = delta_and_sigma_update(X, mu, R.transpose(1,2), v, mu_eta, S, b, a_sigma, b_sigma)
        
        # Update B
        mu, Precision = B_update(X, mu, Precision, mu_eta, S, np_delta, v, C)
        
    return mu, Precision, np_sigma





        
        
        
        
        
        
        
        
        
        
            
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    