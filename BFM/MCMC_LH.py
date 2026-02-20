import torch
import math
import numpy as np
from tqdm import tqdm
from torch.distributions.gamma import Gamma
from BFM.shrinkage import shrinkage
from torch.linalg import solve
from scipy.sparse.linalg import svds
from torch import einsum, eye, randn_like, ones_like, stack

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

def sample_eta(X, B, sigma, device):
    
    _ , r = B.shape
    
    C = (B.T) / sigma
    
    mu = C @ (X / sigma).T
            
    return solve(C @ C.T + eye(r, device = device, dtype = torch.float64), mu + C @ randn_like(X).T + randn_like(mu))

def sample_beta(X, D, eta_sample, sigma2_sample):
    
    r, _ = eta_sample.shape
    
    N, P = X.shape
    
    C = (D.view(P,r,1) * eta_sample.view(1,r, N)) / sigma2_sample.sqrt().view(P,1,1)  
        
    phi = D * (eta_sample @ X / sigma2_sample).T + einsum('bij,bj->bi', C, randn_like(X.T)) + torch.randn(P, r, device = X.device, dtype = X.dtype)
        
    return  D * solve(einsum('bij,bjk->bik', C, C.transpose(1,2)) + eye(r, device = X.device, dtype = X.dtype).view(1,r,r),phi)


def value_b(a, c, r, P, eps1, eps2):
    
    H = 120 / ((a - 1) * (a - 2) * (a - 3) * (a - 4))
    
    h = (eps1 * eps2 / (2 * H)) ** (1 / r)
    
    b = (r + 1) ** c / P ** 0.25 * h ** (r / 4)
    
    return b
    
    
def Gibbs_sampling(X, device, r = 50, a = 10, c = 0.3, M = 5000, burn_in = 5000, score = False):
    
    if a <= 4:
        raise ValueError("a should larger than 4")
    
    if c <= 0.25:
        raise ValueError("c should larger than 0.25")
    
    N,P = X.shape
    
    a_sigma = 1
    b_sigma = 1
    
    b = value_b(a, c, r, P, 1e-1, 1e-1)
    
    ## Initialization
    B_sample, sigma2_estimator = Initialization(X, r)
    
    X = torch.from_numpy(X).to(device).to(torch.float64)
    
    B_sample = torch.from_numpy(B_sample).to(device).to(X.dtype)
    
    sigma2_sample = sigma2_estimator * torch.ones(P, device = device, dtype = X.dtype)
    
    B_samples = []
    sigma2_samples = []
    
    if score == True:
        eta_samples = []
    
    for i in tqdm(range(1, M + burn_in)):
        
        # Sample eta
        eta_sample = sample_eta(X, B_sample, sigma2_sample.sqrt(),device)
        
        # Sample shrinkage parameter
        D = shrinkage(B_sample, a, b, c)
        
        # Sample sigma2
        sigma2_sample = (b_sigma + 0.5 * (X.T - B_sample @ eta_sample).pow(2).sum(1)) / Gamma(a_sigma + 0.5 * N, ones_like(sigma2_sample)).sample()
        
        # Sample B
        B_sample = sample_beta(X, D, eta_sample, sigma2_sample)
            
        if (i + 1) > burn_in:
            
            B_samples.append(B_sample)
            sigma2_samples.append(sigma2_sample)
            
            if score == True:
                eta_samples.append(eta_sample)
                
    if score == True:
        return stack(B_samples).squeeze().to('cpu'), stack(eta_samples).squeeze().to('cpu'), stack(sigma2_samples).squeeze().to('cpu')
    else:
        return stack(B_samples).squeeze().to('cpu'), stack(sigma2_samples).squeeze().to('cpu')




