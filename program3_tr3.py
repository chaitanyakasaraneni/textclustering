#!/usr/bin/env python
# coding: utf-8

# ### CMPE 255 Programming Assignment 3

# In[1]:


import numpy as np
import scipy as sp
import random
import math
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabaz_score
import networkx as nx


# In[2]:


def csr_read(fname, ftype="csr", nidx=1):
    r""" 
        Read CSR matrix from a text file. 
        
        \param fname File name for CSR/CLU matrix
        \param ftype Input format. Acceptable formats are:
            - csr - Compressed sparse row
            - clu - Cluto format, i.e., CSR + header row with "nrows ncols nnz"
        \param nidx Indexing type in CSR file. What does numbering of feature IDs start with?
    """
    
    with open(fname) as f:
        lines = f.readlines()
    
    if ftype == "clu":
        p = lines[0].split()
        nrows = int(p[0])
        ncols = int(p[1])
        nnz = long(p[2])
        lines = lines[1:]
        assert(len(lines) == nrows)
    elif ftype == "csr":
        nrows = len(lines)
        ncols = 0 
        nnz = 0 
        for i in range(nrows):
            p = lines[i].split()
            if len(p) % 2 != 0:
                raise ValueError("Invalid CSR matrix. Row %d contains %d numbers." % (i, len(p)))
            nnz += len(p)/2
            for j in range(0, len(p), 2): 
                cid = int(p[j]) - nidx
                if cid+1 > ncols:
                    ncols = cid+1
    else:
        raise ValueError("Invalid sparse matrix ftype '%s'." % ftype)
    val = np.zeros(int(nnz), dtype=np.float)
    ind = np.zeros(int(nnz), dtype=np.int)
    ptr = np.zeros(nrows+1, dtype=np.long)
    n = 0 
    for i in range(nrows):
        p = lines[i].split()
        for j in range(0, len(p), 2): 
            ind[n] = int(p[j]) - nidx
            val[n] = float(p[j+1])
            n += 1
        ptr[i+1] = n 
    
    assert(n == nnz)
    
    return csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.float)


# In[3]:


def csr_idf(matrix, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        matrix = matrix.copy()
    nrows = matrix.shape[0]
    nnz = matrix.nnz
    ind, val, ptr = matrix.indices, matrix.data, matrix.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else matrix


# In[4]:


def csr_l2normalize(matrix, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        matrix = matrix.copy()
    nrows = matrix.shape[0]
    nnz = matrix.nnz
    ind, val, ptr = matrix.indices, matrix.data, matrix.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = float(1.0/np.sqrt(rsum))
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return matrix


# In[5]:


#Read CSR matrix from the input file
csrMatrix = csr_read('train.dat', ftype="csr", nidx=1)

#Scale the CSR matrix by idf (Inverse Document Frequency)
csrIDF = csr_idf(csrMatrix, copy=True)

#Normalize the rows of a CSR matrix by their L-2 norm.
csrL2Normalized = csr_l2normalize(csrIDF, copy=True)

#Obtain a dense ndarray representation of the CSR matrix.
denseMatrix = csrL2Normalized.toarray()


# In[6]:


print(csrL2Normalized)
csrL2Normalized.shape


# In[7]:


print(denseMatrix.shape)


# In[8]:


kmeans = MiniBatchKMeans(n_clusters=200,random_state = 0)
kmeans.fit(csrL2Normalized)


# In[9]:


label = kmeans.labels_
points =centroids= centers = kmeans.cluster_centers_


# In[10]:


centers.shape, label.shape
indices = np.asarray(list(range(0,8580)))
lab = np.column_stack([indices,label])


# In[11]:


def MyDBSCAN(points, eps,minPts):
    neighborhoods = []
    core = []
    border = []
    noise = []

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
    # Find border points 
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
    # Find noise points
    for i in range(len(points)):
        if i not in core and i not in border:
            noise.append(i)
            
    # # Invoke graph instance to visualize the cluster
    G = nx.Graph()
    nodes = core
    G.add_nodes_from(nodes)
    # Create neighborhood
    for i in range(len(nodes)):
        for p in range(len(nodes)):
            # If the distance is below the threshold, add a link in the graph.
            if p != i and sp.spatial.distance.cosine(points[nodes[i]] ,points[nodes[p]]) <= eps:
                G.add_edges_from([(nodes[i], nodes[p])])
    # List the connected components / clusters
    clusters = list(nx.connected_components(G))
    print("# clusters:", len(clusters))
    print("clusters: ", clusters)
    centers = []
    for cluster in clusters:
        coords = []
        for point in list(cluster):
            coords.append(points[point])
        center = np.mean(coords,axis =0)
        centers.append(center)
    expanded_clusters = clusters
    for pt in border:
        distances = {}
        for i, center in enumerate(centers):
    #         print("point = ", pt, " center = ", i)
    #         print(scipy.spatial.distance.cosine(points[pt],center))
            distances[i] = sp.spatial.distance.cosine(points[pt],center)
    #     distances = 
    #     print("closest cluster for point %d = %d " %(pt, min(distances, key=distances.get)))
        closest_cluster = min(distances, key=distances.get)
        expanded_clusters[closest_cluster].add(pt)
    #     print(clusters[closest_cluster])
    label , centroids, expanded_clusters
    centroid_labels = [len(clusters)+1]* len(centroids)
    for index, clstr in enumerate(expanded_clusters):
        for n in clstr:
            centroid_labels[n]= index
    print(np.unique(centroid_labels))
    final_labels = [0]*len(label)
    for i,l in enumerate(label):
        final_labels[i] = centroid_labels[l]
    np.unique(final_labels)
    return final_labels


# In[13]:


kValues = list()
scores = list()


eps =0.5
minPts = 1
d = MyDBSCAN(points, eps,minPts)
with open("submission1.txt", "w") as f:
            for l in d:
                f.write("%s\n"%l)
# for k in range(3,22,2):
#     clustering = DBSCAN(eps= eps, min_samples=k,metric="euclidean").fit(lab)
#     labels = clustering.labels_    
#     score = calinski_harabaz_score(lab, labels)
#     kValues.append(k)
#     scores.append(score)
#     print ("For K= %d Calinski Harabaz Score is %f" %(k, score))


# In[14]:


# %matplotlib inline
# import matplotlib.pyplot as plt

# plt.plot(kValues, scores)
# plt.xticks(kValues, kValues)
# plt.xlabel('Number of Clusters k')
# plt.ylabel('Calinski and Harabaz Score')
# plt.title('Trend of Average Distance to Centroid/Diameter')
# plt.grid(linestyle='dotted')

# plt.savefig('plot.png')
# plt.show()

