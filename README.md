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
