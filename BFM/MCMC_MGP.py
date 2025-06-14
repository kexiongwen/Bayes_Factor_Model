import torch
from tqdm import tqdm
from torch.distributions.gamma import Gamma
from torch.linalg import solve, svd
from torch import einsum, eye, randn_like, ones_like, stack

def Initialization(X, n, r):
    
    U, S, _ = svd(X.T / (n-1)**0.5)
    
    sigma2_estimator = S[r:].square().sum()/ (n - r)
    
    mu = U[:,0:r] * (S[0:r].square() - sigma2_estimator).sqrt()
    
    return mu, sigma2_estimator


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
    
def Gibbs_sampling(X, r = 50, M = 10000, burn_in = 10000):

    ## set hyperparameters
    a_sigma = 1
    b_sigma = 1
    v = 3
    a1 = 3
    a2 = 3

    N, P = X.shape
    
    X = X.to(torch.float64)
    device = X.device

    B_samples = []
    sigma2_samples = []

    ## initialization

    B_sample, sigma2_sample = Initialization(X, N, r)
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
        C = (D.view(P,r,1) * eta_sample.view(1,r,N))/ sigma2_sample.sqrt().view(P,1,1)  
        
        b = D * (eta_sample @ X / sigma2_sample).T + einsum('bij,bj->bi', C, randn_like(X.T)) + randn_like(B_sample)

        B_sample = D * solve(einsum('bij,bjk->bik', C, C.transpose(1,2)) + eye(r,device = device, dtype = torch.float64).view(1,r,r),b)

        if (i + 1) > burn_in:

            B_samples.append(B_sample)
            sigma2_samples.append(sigma2_sample)

    return stack(B_samples).squeeze().to('cpu'), stack(sigma2_samples).squeeze().to('cpu')