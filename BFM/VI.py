import torch
from tqdm import tqdm
from torch.linalg import solve, solve_triangular, cholesky, inv, svd
from BFM.shrinkage import local_shrinkage
from torch.distributions.gamma import Gamma
from torch import einsum


def Initialization(X, n, r):
    
    U, S, _ = svd(X.T / (n-1)**0.5)
    
    sigma2_estimator = S[r:].square().sum()/ (n - r)
    
    mu = U[:,0:r] * (S[0:r].square() - sigma2_estimator).sqrt()
    
    return mu, sigma2_estimator
    
    
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


def delta_and_sigma_update(X, mu, R, v, mu_eta, Psi, b, c, a_sigma, b_sigma):
    
    P,r = mu.size()
    
    _, n = mu_eta.size()
    
    device = mu.device
    
    np_sigma = torch.zeros(P, device = device, dtype = torch.float64)
    
    np_delta = b
    
    weight = torch.linspace(1, r, steps = r, device = device, dtype = torch.float64).pow(0.25 + c)
    
    for i in range(2):
        
        B_sample = Sample_B(mu, R, v, S = 100)
        
        np_sigma +=  (0.5 * ((X.T).view(P,n,1) - einsum('prs,rn-> pns',  B_sample , mu_eta)).square().sum(1) + 0.5 * n * (B_sample * einsum('prs,rr-> prs', B_sample, Psi)).sum(1) + b_sigma).sum(1) / 200
        
        np_delta +=  ((B_sample.abs().sqrt() * weight.view(1,r,1)).sum((0,1))).sum() / 200 
        
    return np_sigma, (a_sigma + 0.5 * n) / np_sigma, np_delta
        

     
def eta_update(mu_eta, Psi, X, C, mu, Precision, v):
    
    P,r = mu.size()
        
    device = mu.device
    
    muTC = mu.T * C
    
    Psi = torch.eye(r, device = device, dtype = torch.float64) +  muTC @ mu + (C.view(P,1,1) *  inv(Precision)).sum(0) * (v / (v-2))
    
    mu_eta = solve(Psi, muTC @ X.T)
            
    return mu_eta, Psi


def B_update(X, mu, Precision, mu_eta, Phi, np_delta, v, a, C):
    
    P,r = mu.size()
    n,_ = X.size()
    
    device = mu_eta.device
    
    for i in range(1,200):
                
        lr1 =  (1 / n) * (i + 30) ** (-0.75)
        lr2 =  (1 / n) * (i + 30) ** (-0.75)
        
        B_sample, u = Sample_B(mu, cholesky(Precision).transpose(1,2), v, S = 1, extra_output = True)
        
        lam = Gamma(2 * P * r + 5 + a , np_delta).sample()
        
        Lambda = local_shrinkage(lam, B_sample).square()
        
        mu.add_(solve(Precision, C.view(-1,1) * ((mu @ mu_eta - X.T) @ mu_eta.T + n * mu @ Phi) + B_sample * Lambda) / u.view(-1,1), alpha = -lr1)
        
        Precision.mul_(1 - lr2).add_((C.view(-1,1,1) * (mu_eta @ mu_eta.T + n * Phi).view(1,r,r) + torch.diag_embed(Lambda)) / u, alpha = lr2)
        
    return mu, Precision


def NGVI(X, a = 1, b = 100, c = 0.25, v =  1000,  r = 50, a_sigma = 1, b_sigma = 1):
    
    n, P = X.size()
    
    X = X.to(torch.float64)
    
    device = X.device
    
    ## Initialization
    mu, sigma2_estimator = Initialization(X, n, r)
    C = (1 / sigma2_estimator) *  torch.ones(P, device = device, dtype = torch.float64)
    Precision =  1e2 * torch.eye(r, device = device, dtype = torch.float64).repeat(P, 1, 1)
    mu_eta = torch.zeros(r, n, device = device, dtype = torch.float64)
    Psi =  torch.eye(r, device = device, dtype = torch.float64)
    
    for i in tqdm(range(200)):
        
        R = cholesky(Precision)
        
        # Update eta 
        mu_eta, Psi = eta_update(mu_eta, Psi, X, C, mu, Precision, v)
        
        S = inv(Psi)
        
        # Update delta and sigma2 
        np_sigma, C, np_delta = delta_and_sigma_update(X, mu, R.transpose(1,2), v, mu_eta, S, b, c, a_sigma, b_sigma)
        
        # Update B
        mu, Precision = B_update(X, mu, Precision, mu_eta, S, np_delta, v,  a, C)
        
    return mu, Precision, np_sigma  





        
        
        
        
        
        
        
        
        
        
            
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    