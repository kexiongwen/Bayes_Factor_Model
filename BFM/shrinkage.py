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

def shrinkage(param, a, b, c):
    
    P,r = param.size()
    
    device = param.device
    
    # Sample lam
    
    weight = torch.linspace(1, r, steps = r, device = device, dtype = torch.float64).pow(c)
    
    ink = param.abs().sqrt() * weight
    
    lam = Gamma(2 * P + a, ink.sum(0) + b).sample()
    
    ink = ink * lam 
    
    #Sample V
    
    v = 2 / inv_gauss(1 / ink)
    
    #Sample tau
    
    tau = v / inv_gauss(v / ink.square()).sqrt()
    
    if torch.any(torch.isinf(tau)):
        tau[torch.isinf(tau)] = 200
        
    return tau / (weight * lam).square()
