import numpy as np
import warnings
from numpy.linalg import norm
from make_video import make_video
from sklearn.utils.extmath import randomized_svd

# Entire algo is define to solve M  = L + S
# M: original video, flattened into a giant matrix (frames x pixels).
# L: The Low-Rank matrix. This is the background. It's "low-rank" because the background is mostly static, so every frame is very similar to the others.
# S: The Sparse matrix. This is the foreground. It's "sparse" because in any given frame, only a tiny fraction of pixels are moving objects (the rest are just black/zero).
# The algorithm's job is to find the best L and S that add up to M.


# key operation for creating sparse S. It takes a matrix and subtracts the threshold from every single value .. anything that crossed 0( goes -ve) are clipped to zero.
# Algo forces any small noise pixels in foreground down to zero, so only real motion pixels are left. 
def apply_soft_thresh(matrix, threshold):
    "Apply soft thresholding to an array"
    return np.sign(matrix) * np.maximum(np.abs(matrix) - threshold, 0)

# add up the absolute value of every single number in the matrix
# measurement tool; Sparse matrix shud have a low L1 norm
# used to see how sparse current guess for S is 
def calculate_l1_norm(matrix):
    """Calculates the L1 norm of a matrix"""
    return np.abs(matrix).sum()


# Core of how RPCA finds the low-rank matrix ( take noisy and find closest possible simple matrix to it)
# Calculates the Singular Value Decomposition(SVD) or the matrix. 
# findamental components of the matrix: U, s , V 
# U = Left singular vectors (columns are "eigen-backgrounds")
# s: list of singular values that represent the importance of each component (Higher numbers = more important)
# V = Right singular vectors (transposed)
def apply_sv_thresh(X, threshold, num_svalue):
    """
    Perform singular value thresholding.
    
    Parameters: 
    X : array of shape [n_samples, n_features], The input array.
    threshold : float, The threshold for the singualar values.
    num_svalue : int, The number of singular values to compute.
    
    Returns:
    X_thresh : The output after performing singular value thresholding.
    grater_sv : The number of singular values of 'X' which were greater than 'threshold'
    (U, s, V): The singular value decomposition

    """
    # Decompose the matrix 
    U, s, V = randomized_svd(X, num_svalue)

    # count the important component (singular value> threshold)
    greater_sv = np.count_nonzero(s > threshold)

    # apply soft thresholding to the list of singular values , cleans the matrix .. leaving only stable background
    s = apply_soft_thresh(s, threshold)

    # # Convert the cleaned s back into a diagonal matrix, only using the imp (non zero) singular values 
    S = np.diag(s)

    # Reconstruct final matrix by L = U * S * V , this is cleaned bacground
    X_thresh = np.dot(U, np.dot(S, V))

    # X_thresh: The final low-rank matrix (L)
    # greater_sv: The hint for the main loop to speed up next iteration
    # (U, s, V): The components (we need 's' for the cost function)
    return X_thresh, greater_sv, (U, s, V)


def rpca(M, height, width, lam = None, mu = None, max_iter = 1000, eps_primal = 1e-7, eps_dual = 1e-5,
         rho = 1.6, initial_sv = 10, max_mu = 1e6, verbose = False, save_interval = 100):
    """
    Implements the Robust PCA algorithm via Principal Component Pursuit
    The algorithm used for optimization is the "Inexact ALM" method.

    Parameters:
    M : array-like, shape (n_samples, n_features)
        The input matrix.
    lam : float, optional
        The importance given to sparsity. Increasing this parameter will yeild
        a sparser 'S'. If not given it is set to :math:'\\frac{1}{\\sqrt{n}}'
        where "n = max(n_samples, n_features".
    mu : float, optional
        The initial value of the penalty parameter in the Augmented Lagrangian
        Multiplier (ALM) algorithm. This controls how much attention is given
        to the constraint in each iteration of the optimization problem.
    max_iter : int, optional
        The maximum number of iterations the optimization algortihm will run
        for.
    eps_primal : float, optional
        The threshold for the primal error in the convex optimization problem.
        If the primal and the dual error fall below "eps_primal" and
        "eps_dual" respectively, the algorithm converges.
    eps_dual :  float, optinal
         The theshold for the dual error in the convex optimzation problem.
    rho : float, optional
        The ratio of the paramter "mu" between two successive iterations.
        For each iteration "mu" is updated as "mu = mu*rho".
    initial_sv : int, optional
        The number of singular values to compute during the first iteration.
    max_mu : float, optional
        The maximum value that "mu" is allowed to take.
    verbose : bool, optional
        Whether to print convergence statistics during each iteration.
    save_interval : int, optional
        The number of iterations for which the result will be saved.
   
    Returns:
    L : array, shape (n_samples, n_features), The low rank component.
    S : array, shape (n_samples, n_features), The sparse component.
    (U, s, Vt) : tuple of arrays, The singular value decomposition of the "L"
    n_iter : int, The number of iterations taken to converge.
    """

    # Set to default depending on matrix dimensions
    # lam is best at this value 
    if lam is None:
        lam = 1.0/np.sqrt(max(M.shape))

    # Metrics to evaluate for plotting and verification
    metrics = {
        "iterations": [],
        "primal_error": [],
        "dual_error": [],
        "objective_cost": [] 
    }

    # 'd' is the minimum dimension, used for the SVD rank update
    d = min(M.shape)

    # if mu not given, use default 
    # depends on the spectral norm (the largest singular value) of entire original video matrix
    if mu is None:
        mu = 1.25/norm(M, 2)

    # Initialize S (Sparse Foreground) as an all-zero matrix.
    S = np.zeros_like(M)

    # Initialize L (Low-Rank Background) as an all-zero matrix.
    L = np.zeros_like(M)

    # 'Y' is the Dual Variable, or Lagrange Multiplier.
    # It acts as a referee that tracks the error (M - L - S).
    # We normalize M by J to keep Y in a reasonable range.
    J = min(norm(M, 2), np.max(np.abs(M)))
    Y = M/J

    # Pre-calculate the Frobenius norm of M. This is the "total energy"
    # of the original video. Used to normalize the error metrics.
    M_fro_norm = norm(M, 'fro')

    # 'sv' is our guess for the rank. Start with a small number for speed.
    sv = initial_sv

    # Initialize s_vals to avoid errors if loop doesn't run or accesses it early
    s_vals = np.zeros(1)

    # --- Main ALM Optimization Loop ---
    for iter_ in range(max_iter):

        # Store the previous S to calculate the Dual Error (stability)
        S_old = S
        
        # Solve for S
        # Find the Sparse matrix S by shrinking the error term (M - L + Y/mu) using the soft-threshold function.
        fore_thresh = lam/mu
        S = apply_soft_thresh(M - L + (Y/mu), fore_thresh)
        
        # Solve for L
        # Find the Low-Rank matrix L by shrinking the singular values of the remaining error (M - S + Y/mu).
        back_thresh = 1/mu
        L, svp, svd_data = apply_sv_thresh(M - S + (Y/mu), back_thresh, sv)
        
        # get the singular values from the SVD data 
        s_vals = svd_data[1] 
        
        # Update the Dual Variable (Y)
        Y = Y + mu*(M - L - S)

        # Update the penalty (mu)  
        # Foreground threshold is constantly getting smaller (more precise) as the algorithm runs.
        mu_old = mu # store for dual error calc 
        mu = rho*mu # forces algo to focus more on the M = L+S constraint as it gets closer to a soln 
        mu = min(mu, max_mu) # cap the penalty 


        # SVD Rank Heuristic for speed 
        # Update our rank guess ('sv') based on the actual rank ('svp')
        # This makes the next SVD calculation faster.
        if svp < sv:
            sv = svp + 1
        else:
            sv = svp + int(round(0.05*d))

        # Rank can't be > matrix dimensions
        sv = min(sv, M.shape[0], M.shape[1])

        # Calculate Metrics for this iteration
        # Primal Error: "Is M = L + S true?" (Measures reconstruction accuracy)
        primal_error = norm(M - L - S, 'fro')/M_fro_norm
        # Dual Error: "Has the solution stopped changing?" (Measures stability)
        dual_error = mu_old*norm(S - S_old, 'fro')/M_fro_norm

        # Calculate objective cost: Cost = ||L||* + λ * ||S||_1
        # Cost = Sum(Singular Values) + Lambda * Sum(Abs(Sparse))
        nuclear_norm = np.sum(s_vals)   # ||L||* (Sum of singular values)
        l1_norm = np.sum(np.abs(S))     # ||S||_1 (Sum of absolute values)
        current_cost = nuclear_norm + (lam * l1_norm)
        
        # Append to metrics
        metrics["iterations"].append(iter_)
        metrics["primal_error"].append(primal_error)
        metrics["dual_error"].append(dual_error)
        metrics["objective_cost"].append(current_cost)

        # print live stats if verbose is on
        if verbose:
            print(f'Iteration {iter_}: Primal= {primal_error:.6f}, Dual= {dual_error:.6f}, Cost= {current_cost:.2f}')

        # Save intermediate videos for debugging 
        if save_interval and (iter_+1) % save_interval == 0:
            make_video(L, height, width, output_path=f'./output_rpca/background_{iter_}.mp4')
            make_video(S, height, width, output_path=f'./output_rpca/foreground_{iter_}.mp4')

        # check for convergence 
        if primal_error < eps_primal and dual_error < eps_dual:
            break

    # incase of non convergence
    if iter_ == max_iter-1:
        warnings.warn(f'Warning: Failed to converge within {max_iter} iterations')
    
    # Calculate final rank based on the last singular values
    final_rank = np.count_nonzero(s_vals)
    
    # Return the final Background (L), Foreground (S), Rank, and Metrics
    return L, S, final_rank, metrics