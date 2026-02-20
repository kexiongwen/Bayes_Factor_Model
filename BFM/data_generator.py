import numpy as np
import torch
from scipy.stats import uniform
from BFM.MCMC_LH import Gibbs_sampling as GS_LH
from BFM.MCMC_MGP import Gibbs_sampling as GS_MGP
from BFM.MCMC_CSP import Gibbs_sampling as GS_CSP
from BFM.VI import NGVI
from BFM.utils import FDR_FNR_mcmc, FDR_FNR_VI, FDR_FNR_COV


def synthetic_example1(P, K, N):
    
    B_0 = np.random.binomial(1, 1 / 3,(P,K)) * np.random.rand(P,K)
    cov_0 = B_0 @ B_0.T + np.diag(uniform.rvs(loc = 0.1, scale = 0.9, size = P))
    X = np.random.multivariate_normal(np.zeros(P), cov_0, N)
    
    return B_0, cov_0, X


def synthetic_example2(P, K, N):
    
    B_0 = np.zeros((P, K))
    for j in range(K):
        for i in range(364 * j, 364 * j + 500):
            B_0[i, j] = 1
    cov_0 = B_0 @ B_0.T + np.eye(P)
    
    X = np.random.multivariate_normal(np.zeros(P), cov_0, N)
    
    return B_0, cov_0, X


def rv_coefficient(A, B):
    # A and B are covariance matrices (symmetric, PSD)
    tr_AB = np.trace(A @ B)
    tr_A2 = np.trace(A @ A)
    tr_B2 = np.trace(B @ B)
    return tr_AB / np.sqrt(tr_A2 * tr_B2)


def experiment_mcmc(P, K, N, sampler, repetition = 50, example = 2):
    
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    FDRs = []
    FNRs = []
    FNs = []
    RVs = []
    num_factors = []
    
    ESS_means = []
    ESS_medians = []
    ESS_maxs = []
    ESS_mins = []
    
    
    for i in range(repetition):
    
        if example == 1:
            
            B_0, cov_0, X = synthetic_example1(P, K, N)
            
        else:
            
            B_0, cov_0, X = synthetic_example2(P, K, N)
            
        if sampler == 'LH':
            
            B_sample, sigma2_sample = GS_LH(X, device = device1)
        
        elif sampler == 'MGP':
            
            B_sample, sigma2_sample = GS_MGP(X, device = device1)
            
        else:
            
            B_sample, sigma2_sample = GS_CSP(X, device = device1)
            
        ess_mean, ess_median, ess_max, ess_min = ESS(B_sample.numpy())
            
        cov_mcmc = torch.einsum('bij,bjk->ik',B_sample, B_sample.transpose(1,2)) / B_sample.size(0) + torch.diag(sigma2_sample.mean(0))
        
        _, _, num_K_mcmc = FDR_FNR_mcmc(B_sample, B_0)
        
        FN = (torch.from_numpy(cov_0) - cov_mcmc).square().sum().sqrt()
        
        FDR_COV_mcmc, FNR_COV_mcmc = FDR_FNR_COV(cov_0, cov_mcmc.numpy())
        
        RV = rv_coefficient(cov_0, cov_mcmc.numpy())
        
        FDRs.append(FDR_COV_mcmc)
        
        FNRs.append(FNR_COV_mcmc)
        
        FNs.append(FN)
        
        RVs.append(RV)
        
        num_factors.append(num_K_mcmc)
        
        ESS_means.append(ess_mean)
        
        ESS_medians.append(ess_median)
        
        ESS_maxs.append(ess_max)
        
        ESS_mins.append(ess_min)
        
        
    statistic = {"FDR" : np.mean(FDRs),"FDR_std" : np.std(FDRs), "FNR" : np.mean(FNRs), "FNR_std" : np.std(FNRs), \
                 "FN" : np.mean(FNs), "FN_std" : np.std(FNs), "RV" : np.mean(RVs), "RV_std" : np.std(RVs), \
                     "Num_FA" : np.mean(num_factors), "Num_FA_std" : np.std(num_factors)}
    
    ESS = {"ESS_mean" : np.mean()}
    
    return statistic



def experiment_VI(P, K, N, repetition = 50, example = 2):
    
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    FDRs = []
    FNRs = []
    FNs = []
    RVs = []
    num_factors = []
    
    for i in range(repetition):
    
        if example == 1:
            
            B_0, cov_0, X = synthetic_example1(P, K, N)
            
        else:
            
            B_0, cov_0, X = synthetic_example2(P, K, N)
            
        mu, Cov, np_sigma, v = NGVI(X, device = device1)
            
        cov_VI = mu @ mu.T + (v / (v - 2)) * torch.diag(torch.vmap(torch.trace)(Cov)) + torch.diag(np_sigma / (0.5 * N))
        
        _, _, num_K_VI = FDR_FNR_VI(mu, Cov, v, B_0)
        
        FN = (torch.from_numpy(cov_0) - cov_VI).square().sum().sqrt()
        
        FDR_COV_VI, FNR_COV_VI = FDR_FNR_COV(cov_0, cov_VI.numpy())
        
        RV = rv_coefficient(cov_0, cov_VI.numpy())
        
        FDRs.append(FDR_COV_VI)
        
        FNRs.append(FNR_COV_VI)
        
        FNs.append(FN)
        
        RVs.append(RV)
        
        num_factors.append(num_K_VI)
        
    statistic = {"FDR" : np.mean(FDRs),"FDR_std" : np.std(FDRs), "FNR" : np.mean(FNRs), "FNR_std" : np.std(FNRs), \
                 "FN" : np.mean(FNs), "FN_std" : np.std(FNs), "RV" : np.mean(RVs), "RV_std" : np.std(RVs), \
                     "Num_FA" : np.mean(num_factors), "Num_FA_std" : np.std(num_factors)}
    
    return statistic
        
    
    
        
        
        
        
        
        
        
        
        
        
        
        
    
        
    
    