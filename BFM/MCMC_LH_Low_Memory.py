import torch
from tqdm import tqdm
from torch.distributions.gamma import Gamma
from BFM.shrinkage import shrinkage
from torch.linalg import solve
from BFM.PM_SVD import PM_SVD
from torch import einsum, eye, randn_like, ones_like

def Initialization(X, n, r):
        
    U, S = PM_SVD(X.T / (n-1) ** 0.5, n)
    
    sigma2_estimator = S[r : ].square().sum() / (len(S) - r)
    
    mu = U[:,0:r] * (S[0:r].square() - sigma2_estimator).sqrt()
    
    return mu + 1e-2 * torch.randn_like(mu), sigma2_estimator

def sample_eta(X, B, sigma, device):
    
    _ , r = B.shape
    
    C = (B.T) / sigma
    
    mu = C @ (X / sigma).T
            
    return solve(C @ C.T + eye(r, device = device, dtype = X.dtype), mu + C @ randn_like(X).T + randn_like(mu))

def Gibbs_sampling(X, a, b, c = 0.05, r = 50, M = 5000, burn_in = 5000):
    
    N,P = X.size()
    
    a_sigma = 1
    
    b_sigma = 1
    
    X = X.to(torch.float64)
    
    device = X.device
    
    ## Initialization
    
    B_sample, sigma2_sample = Initialization(X, N, r)
    
    sigma2_mean = torch.zeros(P, device = device, dtype = X.dtype)
    B_mean = torch.zeros_like(B_sample)
    count = 0
    
    for i in tqdm(range(1, M + burn_in)):
        
        # Sample eta
        eta_sample = sample_eta(X, B_sample, sigma2_sample.sqrt(),device)
        
        # Sample shrinkage parameter
        D = shrinkage(B_sample, a, b, c)
        
        # Sample sigma2
        sigma2_sample = (b_sigma + 0.5 * (X.T - B_sample @ eta_sample).pow(2).sum(1)) / Gamma(a_sigma + 0.5 * N, ones_like(sigma2_sample)).sample()
        
        # Sample B
        C = (D.view(P,r,1) * eta_sample.view(1,r,N)) / sigma2_sample.sqrt().view(P,1,1)  
        
        phi = D * (eta_sample @ X / sigma2_sample).T + einsum('bij,bj->bi', C, randn_like(X.T)) + randn_like(B_sample)
        
        B_sample = D * solve(einsum('bij,bjk->bik', C, C.transpose(1,2)) + eye(r, device = device, dtype = X.dtype).view(1,r,r),phi)
        
        if (i + 1) > burn_in:
                        
            count += 1
            
            B_mean.mul_(1 - 1 / count).add_(B_sample, alpha = 1 / count)
            sigma2_mean.mul_(1 - 1 / count).add_(sigma2_sample, alpha = 1 / count)
         
    return B_mean.to('cpu'), sigma2_mean.to('cpu')




