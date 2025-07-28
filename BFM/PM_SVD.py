import torch

def PM_SVD(A, k):
    """
    Compute top k singular values and left/right eigenvectors using power iteration with deflation.
    
    Args:
        A: Symmetric square matrix (n x p)
        k: Number of top eigenvalues/eigenvectors to compute
        
    Returns:
        left eigenvectors: Matrix with eigenvectors as columns (n x k)
        singularvalues: Tensor of top k eigenvalues (descending order)
    """

    n,p = A.size()
    singular_val = torch.zeros(k, device = A.device, dtype = A.dtype)
    eigenvectors_left = torch.zeros(n, k, device = A.device, dtype = A.dtype)
    
    # Make a copy of the matrix for deflation
    A_current = A.clone()
    count = 0
    
    for i in range(k):
        
        # Random initialization for each eigenvector
        v = torch.randn(p, device = A.device, dtype = A.dtype) 
        v_old = torch.zeros_like(v)
        
        # Power iteration for current eigenpair
        while (v_old - v).norm(p = float('inf')) > 1e-6:
            
            v_old = v.clone()
            v = A_current.T @ (A_current @ v)
            v = v / torch.norm(v)
        
        # Compute Rayleigh quotient (eigenvalue)
        lam = torch.norm(A @ v)
        singular_val[i] = lam
        
        if i > 0:
            if singular_val[i-1] < singular_val[i]:
                singular_val[i] = 0
                break
            
        count += 1    
        
        u = A_current @ v / lam
        
        # Deflation: Subtract the found component
    
        A_current = A_current - lam * torch.outer(u, v)
        
        # Store eigenvectors
        eigenvectors_left[:,i] = u
    
    return eigenvectors_left[:,0:count], singular_val[0:count]
