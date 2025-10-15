from fast_pytorch_kmeans import KMeans
from sklearn.cluster import KMeans as sk_KMeans
# from customKmeans import kmeans_pytorch
import torch
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def clusterMatrix(w, numCodebooks=4, w_bit=4, numBins_hist=64, numBins=16):
    numRows , numCols = w.shape
    q_max = 2**(w_bit-1)-1
    numBins = torch.range(-q_max , q_max , 2*q_max/numBins_hist)
    y = torch.zeros( [numRows,len(numBins)-1] ).to(device)
    y2 = torch.zeros( [numRows,len(numBins)] ).to(device)
    # y = torch.zeros( [numRows,numBins] )
    # y2 = torch.zeros( [numRows,numBins+1] )

    # for loop implementation
    for i in range(numRows):
        # temp = torch.histogram( w[i].to(torch.float32) , bins=numBins , density=True )
        # y[ i ] = temp.hist
        # y2[ i ] = temp.bin_edges
        temp = torch.histc( w[i].cpu().to(torch.float32) , bins = numBins_hist , min = -q_max , max = q_max ).to(torch.float16) / numCols
        y[ i ] = temp

    # bin_edges = temp.bin_edges
    bin_edges = torch.range(-q_max , q_max , 2*q_max/numBins_hist)

    kmeans = KMeans(n_clusters=numCodebooks, mode='euclidean', verbose=0)
    labels = kmeans.fit_predict(y)
    return labels , kmeans.centroids , bin_edges

def clusterMatrix_scikit(w, numCodebooks=4, w_bit=4, numBins_hist=64, numBins=16):
    numRows , numCols = w.shape
    q_max = 2**(w_bit-1)-1
    numBins = torch.range(-q_max , q_max , 2*q_max/numBins_hist)

    y = torch.zeros( [numRows,len(numBins)-1] ).to(device)
    y2 = torch.zeros( [numRows,len(numBins)] ).to(device)
    # y = torch.zeros( [numRows,numBins] )
    # y2 = torch.zeros( [numRows,numBins+1] )

    # for loop implementation
    for i in range(numRows):
        # temp = torch.histogram( w[i].to(torch.float32) , bins=numBins , density=True )
        # y[ i ] = temp.hist
        # y2[ i ] = temp.bin_edges
        temp = torch.histc( w[i].cpu().to(torch.float32) , bins = numBins_hist , min=-q_max , max=q_max ).to(torch.float16) / numCols
        y[ i ] = temp

    # bin_edges = temp.bin_edges
    bin_edges = torch.range(-q_max , q_max , 2*q_max/numBins_hist)

    kmeans = sk_KMeans(n_clusters=numCodebooks)
    labels = kmeans.fit_predict(y.cpu())

    return labels , torch.from_numpy( kmeans.cluster_centers_ ) , bin_edges

def clusterVector(w, bin_edges, numCodebooks=4, numCentroids=4):
    codeBookCentroids = torch.zeros( [numCodebooks , numCentroids] ).to(device)
    for i in range(numCodebooks):
        binPoints = ( bin_edges[0:-1] + bin_edges[1:] )/2
        temp = repeat_points_by_weight(binPoints, w[i])

        kmeans = KMeans(n_clusters=numCentroids, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(temp)
        codeBookCentroids[i] = kmeans.centroids.squeeze()

    return codeBookCentroids

def clusterVector_scikit(w, bin_edges, numCodebooks=4, numCentroids=4):
    codeBookCentroids = torch.zeros( [numCodebooks , numCentroids] ).to(device)
    for i in range(numCodebooks):
        binPoints = ( bin_edges[0:-1] + bin_edges[1:] )/2
        # kmeans over whole range
        temp = repeat_points_by_weight(binPoints, w[i])
        kmeans = sk_KMeans(n_clusters=numCentroids)
        labels = kmeans.fit_predict(temp)
        temp = kmeans.cluster_centers_.squeeze()

        # heuristic for outliers between center mode and extremes
        total_sum = w[i].sum().item()
        target_sum = 0.8 * total_sum
        center_idx = len(w[i]) // 2
        cum_sum = w[i][center_idx].item()
        left = center_idx - 1
        right = center_idx + 1
        while cum_sum < target_sum and (left >= 0 or right < len(w[i])):
            if left > 0:
                cum_sum += w[i][left].item()
                left -= 1
            if right < len(w[i])-1:
                cum_sum += w[i][right].item()
                right += 1
            if cum_sum >= target_sum:
                break

        print(left)
        if left == -1:
            left = left
        numBins_Extreme = left

        # numClusters_Extreme = max( 2 , min( numCentroids-3 , numCentroids - int(numCentroids*(right-left)/len(w[i])) ) )
        if left < 0.25*len(w[i]):
            numClusters_Extreme = 2
        else:
            numClusters_Extreme = 4


        # kmeans over divided range
        # numClusters_Extreme = 2
        # numBins_Extreme = 3
        binPoints1 = torch.cat( [binPoints[0:numBins_Extreme] , binPoints[-numBins_Extreme:]] )
        w1 = torch.cat( [w[i , 0:numBins_Extreme] , w[i , -numBins_Extreme:]] )
        temp1 = repeat_points_by_weight(binPoints1, w1/w1.sum())
        kmeans1 = sk_KMeans(n_clusters=numClusters_Extreme)
        labels = kmeans1.fit_predict(temp1)
        binPoints2 = binPoints[numBins_Extreme:-numBins_Extreme]
        w2 = w[i , numBins_Extreme:-numBins_Extreme]
        temp2 = repeat_points_by_weight(binPoints2, w2/w2.sum())
        kmeans2 = sk_KMeans(n_clusters=numCentroids-numClusters_Extreme)
        labels = kmeans2.fit_predict(temp2)
        temp = np.concatenate( [kmeans1.cluster_centers_.squeeze() , kmeans2.cluster_centers_.squeeze()] )

        # temp = np.concatenate( [torch.linspace( binPoints[0] , binPoints[numBins_Extreme] , 2 + numClusters_Extreme//2 )[1:1+numClusters_Extreme//2],
        #                         torch.linspace( binPoints[-numBins_Extreme] , binPoints[-1] , 2 + numClusters_Extreme//2 )[1:1+numClusters_Extreme//2],
        #                          kmeans2.cluster_centers_.squeeze()] )
        # temp = np.concatenate( [torch.linspace( binPoints[left] , binPoints[right] , 2 + numCentroids-numClusters_Extreme )[1:-1],
        #                          kmeans1.cluster_centers_.squeeze()] )

        codeBookCentroids[i] = torch.from_numpy( temp )

    return codeBookCentroids    

def repeat_points_by_weight(data, weights, scale=100):
    counts = (weights * scale).round().to(torch.int32)
    repeated_data = torch.cat([data[i].repeat(counts[i], 1) for i in range(len(data))], dim=0)
    return repeated_data


def codeBookQuant(w , numCodebooks=4, w_bit=4, numBins_hist=64, numCentroids=8, debugPath=[], debug=False):
    # labels , codeBooks, bin_edges = clusterMatrix(w)
    # codeBookCentroids = clusterVector(codeBooks, bin_edges, numCentroids=8)
    labels , codeBooks, bin_edges = clusterMatrix_scikit(w , numCodebooks=numCodebooks, w_bit=w_bit, numBins_hist=numBins_hist)
    codeBookCentroids = clusterVector_scikit(codeBooks, bin_edges, numCodebooks=numCodebooks, numCentroids=numCentroids)

    if debug:
        plt.figure()
        colorVecLine = ['b', 'r', 'g', 'y', 'c', 'm', 'k']
        scatterColorVec = ['blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'black']
        for i in range(numCodebooks):
            plt.plot( ( bin_edges[0:-1] + bin_edges[1:] )/2 , codeBooks[i] , colorVecLine[i] )
            plt.scatter( codeBookCentroids[i].cpu().numpy() , torch.ones( [1 , numCentroids] )*torch.max(codeBooks[i]) , color=scatterColorVec[i])
        plt.savefig( debugPath )
        plt.close()
    klDiv = 0
    kl_loss = torch.nn.KLDivLoss(reduction="sum", log_target=True)

    numRows , numCols = w.shape
    for i in range(numRows):
        # removing large errors for debugging
        # if i == 0:
        #     print("here")
        # temp = torch.where( torch.abs( w[i].cpu() - codeBookCentroids[ labels[i] , torch.argmin( torch.cdist(w[i].unsqueeze(dim = -1), codeBookCentroids[labels[i]].half().unsqueeze(dim = -1), p=2) , dim=-1) ].half().cpu() ) > 0.5 )    
        # temp2 = w[i , temp[0]]

        # mapping weights to centroids by MSE
        w[i] = codeBookCentroids[ labels[i] , torch.argmin( torch.cdist(w[i].unsqueeze(dim = -1), codeBookCentroids[labels[i]].half().unsqueeze(dim = -1), p=2) , dim=-1) ].half()
        klDiv = klDiv + kl_loss( torch.nn.functional.log_softmax( codeBooks[labels[i]] ) , 
                        torch.nn.functional.log_softmax( torch.histc( w[i].cpu().to(torch.float32) , bins = numBins_hist , min=-(2**(w_bit-1)-1) , max=2**(w_bit-1)-1 ) ) ) 

        # forcing quantized value with large values to be equal to real weight value
        # w[i , temp[0]] = temp2

        

    return w , klDiv/numRows
