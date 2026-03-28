import torch
from torch.distributions.gamma import Gamma


def GIG(mu):
    
    ink = (1e1 * torch.randn_like(mu)).square() / (1e2 * mu)
    
    a = 1 + 0.5 * (ink - ((ink + 2).square() - 4).sqrt())
    
    return torch.where((1 / (1 + a)) >= torch.rand_like(mu), mu / a, mu * a)    

def inv_gauss(mu):
        
    ink = mu * torch.randn_like(mu).square()
    
    a = 1 + 0.5 * (ink - ((ink + 2).square() - 4).sqrt())
    
    return torch.where((1 / (1 + a)) >= torch.rand_like(mu), mu * a, mu / a)

def shrinkage(param, a, weight1, weight2):
    
    P,_ = param.size()
    
    # Sample lam
    
    ink = param.abs().sqrt() 
    
    lam = Gamma(2 * P + a + weight1, ink.sum(0) + 1 / weight2).sample()
    
    ink = ink.mul_(lam)
    
    # Sample V
    
    v = 2 / inv_gauss(1 / ink)
    
    # Sample tau
    
    tau = v / inv_gauss(v / ink.square()).sqrt()
    
    if torch.any(torch.isinf(tau)):
        tau[torch.isinf(tau)] = 200
        
    return tau / lam.square()
