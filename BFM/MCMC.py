import torch
from tqdm import tqdm
from torch.distributions.gamma import Gamma
from BFM.shrinkage import shrinkage

def sample_eta(X, B_star, sigma, device):
    
    _ , r = B_star.shape
    
    C = (B_star.T) * sigma
    
    mu = C @ (X * sigma).T
            
    return torch.linalg.solve(C @ C.T + torch.eye(r,device = device, dtype = torch.float64), mu + C @ torch.randn_like(X).T + torch.randn_like(mu))

def Gibbs_sampling(X, a, r = 50, M = 10000, burn_in = 15000):
    
    N,P = X.size()
    
    w = 1
    
    X = X.to(torch.float64)
    
    device = X.device
    
    ## Initialization
    
    B_samples = []
    sigma2_samples = []
    
    B_sample = torch.ones(P, r, device = device, dtype = torch.float64)
    sigma2_sample = torch.ones(P, device = device, dtype = torch.float64)
    
    for i in tqdm(range(1, M + burn_in)):
        
        # Sample eta
        eta_sample = sample_eta(X, B_sample, sigma2_sample.sqrt(),device)
        
        # Sample shrinkage parameter
        D = shrinkage(B_sample, a, 0.05)
        
        # Sample sigma2
        sigma2_sample = 0.5 * (w + (X.T - B_sample @ eta_sample).pow(2).sum(1)) / Gamma(0.5 * (w + N), torch.ones(P, device = device, dtype = torch.float64)).sample()
        
        # Sample B
        C = (D.unsqueeze(-1) * eta_sample.unsqueeze(0))/ sigma2_sample.sqrt().unsqueeze(-1).unsqueeze(-1)       
        
        b = D * (eta_sample @ X / sigma2_sample).T + torch.einsum('bij,bj->bi', C, torch.randn(P, N, device = device, dtype = torch.float64)) + torch.randn(P,r, device = device, dtype = torch.float64)
        
        B_sample = D * torch.linalg.solve(torch.einsum('bij,bjk->bik', C, C.transpose(1,2)) + torch.eye(r,device = device, dtype = torch.float64).unsqueeze(0).repeat(P, 1, 1),b)
            
        if (i + 1) > burn_in:
            
            B_samples.append(B_sample)
            sigma2_samples.append(sigma2_sample)

    return torch.stack(B_samples).squeeze().to('cpu'), torch.stack(sigma2_samples).squeeze().to('cpu')




