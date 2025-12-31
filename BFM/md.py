import torch
from torch import polygamma

def auxiliary_variables(t,z):
    
    D = t / 2 + (2 / 3)
    
    s = (4.5 * t + 6).sqrt()
    
    w = 1 + z / s
    
    return D, s, w

def log_pdf_proposal(t,z):
    
    D, s, w = auxiliary_variables(t,z)
    
    u = D * w.pow(3)
    
    return s, D * 3 * w.log() - u + D, u

def proposal(t, z):
    
    prop = torch.randn_like(t)
    
    s, g, u = log_pdf_proposal(t, prop)
    
    accept = (prop > -s) * (torch.rand_like(prop).log() <= g + prop.square() / 2)
    
    z[accept] = prop[accept]
    
    return z, accept, u

def AR_sample(t):
    
    flag = torch.ones_like(t,dtype = torch.bool)
    
    z = torch.zeros_like(t)
    
    u = torch.ones_like(t)
    
    while flag.sum() > 0:
        
        z_ink, accept, u_ink = proposal(t[flag],z[flag])
        
        z[flag] = z_ink
        
        u[flag] = u_ink
        
        flag[flag.clone().detach()] = ~accept
        
    return z, u


def sample_t(t, u, R, mu):
    
    P,r,_ = R.size()
    
    noise_term = torch.bmm(R, torch.randn(P,r,1, device = t.device, dtype = torch.float64)).view(P,r)
    
    h = mu + ((t + 2) / (2 * u)).sqrt().view(P,1) * noise_term
    
    return h, noise_term

def log_pdf_prior(h, a, b, c):
    
    p,r = h.size()
    
    device = h.device
    
    weight = torch.linspace(1, r, steps = r, device = device, dtype = torch.float64).pow(0.25 + c)
    
    ink = (h.abs().sqrt().mul_(weight)).sum() + b
    
    return  - (2 * p * r + a) * ink.log(), ink 
    

def pathwise_gradient(h, u, t, grad_T, ink, noise_term, a, c):
    
    device = noise_term.device
    
    p, r = noise_term.size()
    
    const = - 0.5 * (p * r + a / 2) / ink
    
    weight = torch.linspace(1, r, steps = r, device = device, dtype = torch.float64).pow(0.25 + c)
    
    term1 = (weight * noise_term / h.abs().sqrt()).sum(1)
        
    term2 = (1 - (t + 2) / u * grad_T) / (2 * (t + 2) * u).sqrt()
    
    return const * term1 * term2


    
def GRT_gradient(t, mu, R, a, b, c):
    
    z, u = AR_sample(t)
    
    D, s, w = auxiliary_variables(t,z)
    
    grad_T = 0.5 * w.square() * (1 - 0.5 * z / s)
    
    h, noise_term = sample_t(t, u, R, mu)
    
    score = 0.5 * u.log() + (0.5 * t / u - 1) * grad_T + 0.5 / D - 2.25 / s.square() - 4.5 * z / (w  * s.pow(3)) - 0.5 * torch.digamma(0.5 * (t + 2))
    
    lpp, ink = log_pdf_prior(h, a, b, c)
    
    # Score-function gradient
    grad_corr = score * lpp
    
    # Pathwise gradient
    grad_pathwise = pathwise_gradient(h, u, t, grad_T, ink, noise_term, a, c)
    
    #print(grad_pathwise)
    
    return grad_pathwise + grad_corr
    #return grad_pathwise
    
    
def gradient_t(t, A, mu, R, a, b, c):
    
    _, r = mu.size()
    
    return A / t.square() + r / (2 * (t + 2)) + 0.25 * (t + 2 + r) * (polygamma(1, (t + 2 + r) / 2) - polygamma(1, (t + 2) / 2)) + GRT_gradient(t, mu, R, a, b, c)


def mirror_descent(v, A, mu, R, a, b, c, lr):
    
    t = v - 2
    
    for i in range(100):
        
        grad = gradient_t(t, A, mu, R, a, b, c)
        
        t = t * torch.exp(lr * grad)
        
    return t + 2
    
    
    
    
    
    
    
    
    
    