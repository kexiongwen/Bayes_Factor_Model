import torch
import arviz as az
import numpy as np
from scipy.stats import t


def check_marginal_CI_VI(mu, Var, v, level):
    
    a = 1 - level
    
    upp = t.ppf(1 - 0.5 * a, df = v, loc = mu, scale = np.sqrt(Var))
    low = t.ppf(0.5 * a, df = v, loc = mu, scale = np.sqrt(Var))
    
    return (0 <= upp) * (0 >= low)


def check_marginal_HDI_mcmc(sampler, level):
    
    # Add chain dimension: (1, N, p, r)
    samples_with_chain = sampler[None, :, :, :]  # or samples[np.newaxis, ...]

    # Define coords and dims
    dims = ["param_dim_0", "param_dim_1"]
    coords = {
        "param_dim_0": np.arange(sampler.shape[1]),  # 0..p-1
        "param_dim_1": np.arange(sampler.shape[2]),  # 0..r-1
    }

    # Create InferenceData
    idata = az.from_dict(
        posterior={"theta": samples_with_chain},
        dims={"theta": dims},
        coords=coords,
    )

    # Compute HPD
    hdi_intervals = az.hdi(idata, var_names=["theta"], hdi_prob = level)
    intervals = hdi_intervals["theta"].values
    
    return (0 <= intervals[:,:,1]) * (0 >= intervals[:,:,0])
    

def FDR_FNR_mcmc(B_sample, B_0):
    
    P, r_0 = B_0.shape
    r = B_sample.size(2)
    B_sample = B_sample.numpy()
    
    B_0 = np.concatenate((B_0, np.zeros((P,r - r_0))), axis = 1)
    
    zero = check_marginal_HDI_mcmc(B_sample, 0.95)
    
    TP = np.sum((zero == False) & (B_0 != 0))
    FP = np.sum((zero == False) & (B_0 == 0))
    FN = np.sum((zero == True) &  (B_0 != 0))
    
    FDR = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    
    num_K = ((P - zero.sum(0)) != 0).sum()

    return FDR, FNR, num_K


def FDR_FNR_VI(mu, Cov, v, B_0):
    
    P, r_0 = B_0.shape
    r = mu.size(1)
    
    B_0 = np.concatenate((B_0, np.zeros((P,r - r_0))), axis = 1)
    Var = torch.diagonal(Cov, dim1=-2, dim2=-1)
    
    Var = Var.numpy()
    mu = mu.numpy()
    v = v.numpy().reshape(-1,1)
    
    zero = check_marginal_CI_VI(mu, Var, v, 0.95)

    TP = np.sum((zero == False) & (B_0 != 0))
    FP = np.sum((zero == False) & (B_0 == 0))
    FN = np.sum((zero == True) &  (B_0 != 0))
    
    FDR = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    
    num_K = ((P - zero.sum(0)) != 0).sum()

    return FDR, FNR, num_K


def FDR_FNR_COV(Sigma_true, Sigma_hat):
    """
    Compute FDR and FNR using only the strict upper triangular part (off-diagonal).
    
    Parameters:
    - Sigma_true: true covariance matrix (n x n)
    - Sigma_hat: estimated covariance matrix (n x n)
    - tau: threshold to declare an entry non-zero in Sigma_hat
    
    Returns:
    - FDR, FNR
    """
    n = Sigma_true.shape[0]
    assert Sigma_true.shape == Sigma_hat.shape == (n, n), "Matrices must be square and same size"
    
    # Get indices of strict upper triangle (i < j)
    iu = np.triu_indices(n, k=1)  # k=1 => exclude diagonal
    
    # Extract values
    true_vals = Sigma_true[iu]
    hat_vals  = Sigma_hat[iu]
    
    # Define supports (boolean arrays over upper triangle only)
    # Use a small tolerance for true matrix to handle numerical noise
    supp_hat  = np.abs(hat_vals) > 1e-4
    supp_true = true_vals != 0
    
    # Counts
    TP = np.sum(supp_true & supp_hat)
    FP = np.sum(~supp_true & supp_hat)
    FN = np.sum(supp_true & ~supp_hat)
    
    # FDR: among all discoveries (TP + FP), how many are false?
    FDR = FP / (FP + TP) if (FP + TP) > 0 else 0.0
    
    # FNR: among all true edges (TP + FN), how many were missed?
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    
    return FDR, FNR


def ESS(samples):
    
    if samples.ndim == 3:
        samples = samples[None, :, :, :]
    
    else:
        samples = samples[None, :, :]
        
    dataset = az.convert_to_dataset(
    samples,
    group = "posterior",          # optional but good practice
    coords = None,                # auto-generated   # labels for P and r dimensions (optional)
    )
    
    ess = az.ess(dataset)
    
    return np.array(ess.to_array()).squeeze()    
    