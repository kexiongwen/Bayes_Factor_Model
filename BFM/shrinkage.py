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
    
    weight = torch.linspace(1, r, steps = r, device = device, dtype = torch.float64).pow(0.25 + c)
    
    ink = param.abs().sqrt() * weight
    
    lam = Gamma(2 * P * r + 5 + a, ink.sum() + b).sample()
    
    ink = ink * lam 
    
    #Sample V
    
    v = 2 / inv_gauss(1 / ink)
    
    #Sample tau
    
    tau = v / inv_gauss(v / ink.square()).sqrt()
    
    if torch.any(torch.isinf(tau)):
        print('inf')
        tau[torch.isinf(tau)] = 100
        
    return tau / (weight * lam).square()


def local_shrinkage(lam, param, c = 0.05):
    
    _,r = param.size()
    
    device = param.device
    
    weight = torch.linspace(1, r, steps = r, device = device, dtype = torch.float64).pow(0.25 + c) * lam
    
    ink = param.abs().sqrt() * weight 
    
    # Sample V
    v = 2 / inv_gauss(1 / ink)
    #v = 2 * GIG(ink)
    
    # Sample tau
    tau = v / inv_gauss(v / ink.square()).sqrt()
    #tau = v * GIG(ink.square() / v).sqrt()
    
    if torch.any(torch.isinf(tau)):
        #print('inf')
        #print(torch.isinf(tau).sum())
        tau[torch.isinf(tau)] = 200
        
    if torch.any(torch.isnan(tau)):
        print('nan')
        print(torch.isnan(tau).sum())
        
    #return  2 * weight.square() / tau
        
    return torch.where(torch.isnan(tau), 1e2,  2 * weight.square() / tau)