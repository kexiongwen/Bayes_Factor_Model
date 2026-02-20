import torch
import numpy as np
from tqdm import tqdm
from torch.distributions.gamma import Gamma
from torch.linalg import solve
from torch import einsum, eye, randn_like, ones_like, stack
from scipy.sparse.linalg import svds

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


def sample_eta(X, B, sigma ,device):
    
    _, r = B.shape

    C = B.T / sigma

    mu = C @ (X / sigma).T
    
    return torch.linalg.solve(C @ C.T + torch.eye(r, device = device, dtype = torch.float64), mu + C @ torch.randn_like(X).T + torch.randn_like(mu))

def sample_delta(B, delta, tau, a1, a2):
    
    p,r = B.shape
    
    ink = (B.square() * tau).sum(0)
    
    for j in range(0, r):
        
        lam = delta.cumprod(0)
    
        if j == 0:
            delta[j] = Gamma(a1 + 0.5 * p * (r - j),  0.5 * ((lam[j:] / delta[j]) * ink [j:]).sum() + 1).sample()
        else:
            delta[j] = Gamma(a2 + 0.5 * p * (r - j),  0.5 * ((lam[j:] / delta[j]) * ink [j:]).sum() + 1).sample()
                
    return delta

def sample_beta(X, D, eta_sample, sigma2_sample):
    
    r, _ = eta_sample.shape
    
    N, P = X.shape
    
    C = (D.view(P,r,1) * eta_sample.view(1,r, N)) / sigma2_sample.sqrt().view(P,1,1)  
        
    phi = D * (eta_sample @ X / sigma2_sample).T + einsum('bij,bj->bi', C, randn_like(X.T)) + torch.randn(P, r, device = X.device, dtype = X.dtype)
        
    return  D * solve(einsum('bij,bjk->bik', C, C.transpose(1,2)) + eye(r, device = X.device, dtype = X.dtype).view(1,r,r),phi)
    
def Gibbs_sampling(X, device, r = 50, M = 5000, burn_in = 5000, score = False):

    ## set hyperparameters
    a_sigma = 1
    b_sigma = 1
    
    v = 3
    a1 = 3
    a2 = 3

    N, P = X.shape
    
    B_samples = []
    sigma2_samples = []
    
    if score == True:
        eta_samples = []

    ## initialization

    B_sample, sigma2_estimator = Initialization(X, r)
    
    X = torch.from_numpy(X).to(device).to(torch.float64)
    
    B_sample = torch.from_numpy(B_sample).to(device).to(X.dtype)
    
    sigma2_sample = sigma2_estimator * torch.ones(P, device = device, dtype = X.dtype)
    

    delta_sample = torch.ones(r, device = device, dtype = torch.float64)
    lam_sample = torch.cumprod(delta_sample, dim = 0)

    for i in tqdm(range(1, M + burn_in)):
        
        # Sample eta
        eta_sample = sample_eta(X, B_sample, sigma2_sample.sqrt(), device)

        # Sample sigma2
        sigma2_sample = (b_sigma + 0.5 * (X.T - B_sample @ eta_sample).pow(2).sum(1)) / Gamma(a_sigma + 0.5 * N , ones_like(sigma2_sample)).sample()

        # Sample local shrinkage parameter
        tau_sample = Gamma(0.5 * (v + 1), 0.5 * (v + B_sample.square() * lam_sample)).sample()

        # Sample global shrinkage parameter
        delta_sample = sample_delta(B_sample, delta_sample, tau_sample, a1, a2)
        lam_sample = torch.cumprod(delta_sample, dim = 0)

        D = 1 / (tau_sample * lam_sample).sqrt()

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