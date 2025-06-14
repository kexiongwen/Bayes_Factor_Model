import torch
from tqdm import tqdm
from torch import multinomial, bincount, einsum, eye, randn_like, ones_like, stack
from torch.distributions.gamma import Gamma
from torch.distributions import StudentT, Normal, Beta
from torch.linalg import solve, svd


def generate_Table(B, C):
    
    H = len(B)
    
    # Create lower triangular mask (i >= j)
    mask = torch.tril(torch.ones(H, H, dtype=torch.bool, device=B.device))
    
    # Fill lower part with B[i], upper part with C[j]
    A = torch.where(mask, B.view(-1, 1), C.view(1, -1))
    
    return A


def Initialization(X, n, r):
    
    U, S, _ = svd(X.T / (n-1)**0.5)
    
    sigma2_estimator = S[r:].square().sum()/ (n - r)
    
    mu = U[:,0:r] * (S[0:r].square() - sigma2_estimator).sqrt()
    
    return mu, sigma2_estimator


def sample_eta(X, B, sigma, device):
    
    _ , r = B.shape
    
    C = (B.T) / sigma
    
    mu = C @ (X / sigma).T
            
    return solve(C @ C.T + eye(r, device = device, dtype = torch.float64), mu + C @ randn_like(X).T + randn_like(mu))


def shrinkage(B, V, alpha, a_theta, b_theta, theta_inf):
    
    _, r = B.shape
    device = B.device
    
    ink = B.square()
    normal_d = Normal(0, theta_inf)
    studentt_d = StudentT(2 * a_theta, 0, b_theta / a_theta)
    J = torch.arange(1, r + 1, device = device, dtype = torch.float64)
    
    log_pdf_N = normal_d.log_prob(B).sum(0)
    log_pdf_T =  studentt_d.log_prob(B).sum(0)
    
    for i in range(20):
        
        w = (1 - V).log().cumsum(0)
        w[0:-1] = V[0:-1].log() + w[0:-1] - (1 - V[0:-1]).log()
        Table = generate_Table(log_pdf_N, log_pdf_T) + w.view(1,-1)
        Z = multinomial((Table - torch.max(Table,1)[0].view(-1,1)).exp(), num_samples = 1).view(-1)
        V = Beta((1 + bincount(Z, minlength = r + 1)[1 : r + 1]).double(), (alpha + (Z.unsqueeze(1) > J).sum(0)).double()).sample() 
        V[-1] = 0        
            
    D = theta_inf * ones_like(B)
    D[:,Z > J] = 1 / Gamma((a_theta + 0.5), b_theta + 0.5 * ink[: , Z > J]).sample().sqrt()
    
    return D, V

def Gibbs_sampling(X, r = 50, M = 10000, burn_in = 10000):
    
    N,P = X.size()
    
    alpha = 5
    a_theta = 2
    b_theta = 2
    theta_inf = 0.2
    
    a_sigma = 1
    b_sigma = 1
    
    X = X.to(torch.float64)
    
    device = X.device
    
    ## Initialization
    B_samples = []
    sigma2_samples = []
    B_sample, sigma2_sample = Initialization(X, N, r)
    V = Beta(torch.ones(r, device = device, dtype = torch.float64), alpha).sample()
    V[-1] = 0
        
    for i in tqdm(range(1, M + burn_in)):
        
        # Sample eta
        eta_sample = sample_eta(X, B_sample, sigma2_sample.sqrt(),device)
        
        # Sample shrinkage parameter
        D, V = shrinkage(B_sample, V, alpha, a_theta, b_theta, theta_inf)
        
        # Sample sigma2
        sigma2_sample = (b_sigma + 0.5 * (X.T - B_sample @ eta_sample).pow(2).sum(1)) / Gamma(a_sigma + 0.5 * N, ones_like(sigma2_sample)).sample()
        
        # Sample B
        C = (D.view(P,r,1) * eta_sample.view(1,r,N)) / sigma2_sample.sqrt().view(P,1,1)  
        
        phi = D * (eta_sample @ X / sigma2_sample).T + einsum('bij,bj->bi', C, randn_like(X.T)) + randn_like(B_sample)
        
        B_sample = D * solve(einsum('bij,bjk->bik', C, C.transpose(1,2)) + eye(r, device = device, dtype = torch.float64).view(1,r,r),phi)
            
        if (i + 1) > burn_in:
            
            B_samples.append(B_sample)
            sigma2_samples.append(sigma2_sample)

    return stack(B_samples).squeeze().to('cpu'), stack(sigma2_samples).squeeze().to('cpu')