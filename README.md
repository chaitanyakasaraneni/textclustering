# Text Clustering
### Description
In this program, we will be implementing a variation of DBSCAN Clustering Algorithm from the scratch. For this, there are two steps involved:
1. We'll cluster the input data using Sklearn's clustering methods such that we obtain > 100 clusters.
2. In the second step, we'll implement a variation of the DBSCAN clustering algorithm that takes input as the clusters,
rather than individual points. You can use any method to compute inter-cluster distances to figure out which points are core, border, or noise points

After that, assign each of the instances in the input data to K clusters identified from 1 to K. All objects in the training data set must be assigned to a cluster. Thus, you can either assign all noise points to cluster K+1 or apply post-processing
after DBSCAN and assign noise points to the closest cluster.

### Data Description
Input data (provided as training data) consists of 8580 text records in sparse format. No labels are provided.

### My Implementation for the DBSCAN
For the first step, I used the MiniBatchKMeans from sklearn.cluster.
```
from sklearn.cluster import MiniBatchKMeans


kmeans = MiniBatchKMeans(n_clusters=200,random_state = 0)
kmeans.fit(csrL2Normalized)
```
In the second step, I implemented my own version of DBSCAN, which is not so effificient and has an NMI (Normalized Mutual
Information Score) of 0.4223

```
# Finding Core points
for i in range(len(points)):
        neighbors = []
        for p in range(0, len(points)):
            # If the distance is below eps, p is a neighbor
            if sp.spatial.distance.cosine(points[i] ,points[p]) <= eps:
                neighbors.append(p)
        neighborhoods.append(neighbors)
        # If neighborhood has at least minPts, i is a core point
        if len(neighbors) >= minPts :
            core.append(i)
            
# Finding Border Points
for i in range(len(points)):
        neighbors = neighborhoods[i]
        # Look at points that are not core points
        if len(neighbors) < minPts:
            for j in range(len(neighbors)):
                # If one of its neighbors is a core, it is also in the core point's neighborhood, 
                # thus it is a border point rather than a noise point
                if neighbors[j] in core:
                    border.append(i)
                    # Need at least one core point...
                    break
# Finding noise points
for i in range(len(points)):
    if i not in core and i not in border:
        noise.append(i)   
```
**Note:** 
- You can use any implementation to get the better NMI and NMI is an external evaluation metric, you can use any internal evaluation metric such as  calinski_harabaz_score from sklearn.metrics.
- In the bisecting k-means, I commented out the last part purposefully. While running that part of the code, make sure you close all other programs/applications as it may return a Memory Error.
