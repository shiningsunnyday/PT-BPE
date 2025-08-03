from copy import deepcopy
import numpy as np
from typing import Optional 


def kabsch(P, Q):
    """
    Perform the Kabsch algorithm to find the optimal rotation matrix R and translation t
    that aligns Q to P.
    
    Parameters:
      P: numpy.ndarray of shape (N, 3)
         Reference coordinates.
      Q: numpy.ndarray of shape (N, 3)
         Coordinates to be aligned.
    
    Returns:
      Q_aligned: numpy.ndarray of shape (N, 3)
         Q after applying the optimal rotation and translation.
      R: numpy.ndarray of shape (3, 3)
         Optimal rotation matrix.
      t: numpy.ndarray of shape (3,)
         Optimal translation vector.
    """
    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    # Center the points
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Compute covariance matrix
    H = np.dot(Q_centered.T, P_centered)
    
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    
    # Compute rotation matrix
    d = np.linalg.det(np.dot(V, U.T))
    D = np.eye(3)
    if d < 0:
        D[2, 2] = -1  # Reflection correction
    R = np.dot(V, np.dot(D, U.T))
    
    # Compute translation
    t = centroid_P - np.dot(centroid_Q, R)
    
    # Apply rotation and translation to Q
    Q_aligned = np.dot(Q, R) + t
    
    return Q_aligned, R, t

def compute_rmsd(P, Q):
    """
    Compute the RMSD between two sets of coordinates P and Q after optimal alignment using the Kabsch algorithm.
    
    Parameters:
      P: numpy.ndarray of shape (N, 3)
         Reference coordinates.
      Q: numpy.ndarray of shape (N, 3)
         Coordinates to be aligned.
    
    Returns:
      rmsd: float
         The root-mean-square deviation after alignment.
    """
    Q_aligned, R, t = kabsch(P, Q)
    diff = P - Q_aligned
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

def compute_value_ranges(strucs):
    keys = list(strucs[0])
    value_ranges = {}
    for k in keys:
        value_ranges[k] = [struc[k] for struc in strucs]
        value_ranges[k] = (min(value_ranges[k]), max(value_ranges[k]))
    return value_ranges

def initialize(strucs, k):
    """
    Interpolate between angular range
    Find angle-specific endpoints
    """
    value_ranges = compute_value_ranges(strucs)
    new_strucs = []
    for i in range(k):
        new_struc = deepcopy(strucs[0])
        for key in new_struc:
            start, end = value_ranges[key]
            new_struc[key] = np.random.uniform(start, end)
        new_strucs.append(new_struc)
    return new_strucs

def k_means(strucs, k, max_iterations=10, tol=0.1):
    """
    Input: 
        D = {d1, d2, ..., dN}         # Set of N protein conformations
        k                             # Number of clusters
        average_structure(cluster)    # Function that returns the average structure of a cluster
        distance(a, b)                # Function that computes RMSD (or similar) between structures a and b
        max_iterations                # Maximum iterations (optional)
        tol                           # Convergence tolerance (optional)

    Output:
        medoids = {m1, m2, ..., mk}   # k representative "average" structures (cluster centers)
        assignments = {a1, a2, ..., aN}  # For each conformation, the index of its assigned cluster

    Algorithm:
    1. Initialization:
        Randomly select k conformations from D as initial centers: medoids = {m1, m2, ..., mk}
        
    2. For iter in 1 to max_iterations:
        a. Assignment Step:
        For each conformation d in D:
            Compute distances: for j from 1 to k, cost[j] = distance(d, m_j)
            Assign d to cluster j* = argmin(cost)
        Let assignments[i] be the cluster index for d_i.
        
        b. Update Step:
        For each cluster j from 1 to k:
            Let cluster_j = { d in D such that d is assigned to j }
            Compute new center: new_m_j = average_structure(cluster_j)
            
        c. Convergence Check:
        If the change in centers (for example, sum_j distance(m_j, new_m_j)) < tol:
            Break the loop.
        Otherwise, set medoids = { new_m_1, new_m_2, ..., new_m_k }

    3. Return medoids, assignments
    """
    medoids = initialize(strucs, k)
    assignments = np.zeros(len(strucs))
    for _iter in range(1, max_iterations+1):
        struc_arr = [[] for _ in range(k)]
        for i, d in enumerate(strucs):
            costs = [distance(d, medoids[j]) for j in range(k)]
            j = np.argmin(costs)
            assignments[i] = j
            struc_arr[j].append(d)
        new_mediods = [average_structure(struc_arr[j]) for j in range(k)]
        diff = sum([distance(m1, m2) for m1, m2 in zip(medoids, new_mediods)])
        if diff < tol:
            break
    return medoids, assignments

def k_medoids(
    strucs,
    k,
    max_iterations: int = 10,
    tol: float = 1e-4,
    *,
    max_num_strucs: Optional[int] = None,
    seed: Optional[int] = None,          
):
    """
    Perform k-medoids clustering on a set of structures.

    ...
    max_num_strucs : int, optional
        If len(strucs) is larger than this value, run the algorithm only on a
        random subset (size = max_num_strucs) and afterwards assign every
        structure to the nearest medoid.
    seed : int or None, optional
        Random seed used for deterministic subsampling and medoid selection.
    """
    
    N = len(strucs)
    k = min(k, N)
    if max_num_strucs is not None and max_num_strucs < k:
        raise ValueError("`max_num_strucs` must be >= k")

    rng = np.random.default_rng(seed)

    # Decide which structures participate in the iterative part
    if max_num_strucs is not None and N > max_num_strucs:
        active_idx = rng.choice(N, size=max_num_strucs, replace=False)
    else:
        active_idx = np.arange(N)

    # --- initialisation (unchanged except we draw from `active_idx`) -----------
    medoid_indices = rng.choice(active_idx, size=k, replace=False)
    medoids = [strucs[i] for i in medoid_indices]

    assignments = np.zeros(N, dtype=int)
    for iteration in range(max_iterations):
        # -------- assignment step (only on the active subset) -----------------
        for i in active_idx:
            costs = [compute_rmsd(strucs[i], medoids[j]) for j in range(k)]
            assignments[i] = np.argmin(costs)

        # -------- update step --------------------------------------------------
        total_shift = 0.0
        new_medoids, new_medoid_indices = [], []
        for j in range(k):
            cluster_members = [idx for idx in active_idx if assignments[idx] == j]
            if not cluster_members:
                new_idx = rng.choice(N)              # re-initialise from *all* data
            else:
                # pick candidate with minimal total intra-cluster distance
                dists = [
                    sum(compute_rmsd(strucs[c], strucs[m]) for m in cluster_members)
                    for c in cluster_members
                ]
                new_idx = cluster_members[int(np.argmin(dists))]
            shift = compute_rmsd(medoids[j], strucs[new_idx])
            total_shift += shift
            new_medoids.append(strucs[new_idx])
            new_medoid_indices.append(new_idx)

        medoids, medoid_indices = new_medoids, new_medoid_indices
        if total_shift < tol:
            print(f"Converged in {iteration + 1} iterations with total shift {total_shift:.6f}.")
            break

    # -------- final assignment for *all* structures ---------------------------
    for i, d in enumerate(strucs):
        costs = [compute_rmsd(d, strucs[idx]) for idx in medoid_indices]
        assignments[i] = np.argmin(costs)

    return medoid_indices, assignments