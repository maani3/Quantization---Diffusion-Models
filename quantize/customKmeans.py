import torch

def kmeans_pytorch(data, K, max_iters=100, tol=1e-4, verbose=False, seed=None):
    """
    Standard K-Means clustering using PyTorch.

    Args:
        data (Tensor): shape (N, D) - input data
        K (int): number of clusters
        max_iters (int): maximum number of iterations
        tol (float): convergence threshold
        verbose (bool): print iteration info
        seed (int or None): random seed for reproducibility

    Returns:
        centroids (Tensor): shape (K, D)
        assignments (Tensor): shape (N,)
    """
    if seed is not None:
        torch.manual_seed(seed)

    N, D = data.shape

    # Initialize centroids by choosing K random data points
    indices = torch.randperm(N)[:K]
    centroids = data[indices].clone()  # shape (K, D)

    for i in range(max_iters):
        # Step 1: Compute distances between each point and each centroid
        # Output shape: (N, K)
        dists = torch.cdist(data, centroids, p=2)

        # Step 2: Assign each point to the closest centroid
        assignments = torch.argmin(dists, dim=1)  # shape (N,)

        # Step 3: Compute new centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            cluster_points = data[assignments == k]
            if cluster_points.shape[0] > 0:
                new_centroids[k] = cluster_points.mean(dim=0)
            else:
                # If a cluster got no points, reinitialize randomly
                new_centroids[k] = data[torch.randint(0, N, (1,))]

        # Step 4: Check for convergence
        shift = (centroids - new_centroids).norm()
        centroids = new_centroids
        if verbose:
            print(f"Iter {i+1}: shift = {shift:.6f}")
        if shift < tol:
            break

    return centroids, assignments
