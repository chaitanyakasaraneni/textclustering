
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle
from sklearn.metrics import calinski_harabaz_score


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
    ptr = np.zeros(int(nrows+1), dtype=np.long)
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


def initialCentroids(matrix):
    matrixShuffled = shuffle(matrix, random_state=0)
    return matrixShuffled[:2,:]


# In[6]:


def similarity(matrix, centroids):
    similarities = matrix.dot(centroids.T)
    return similarities


# In[7]:


def findClusters(matrix, centroids):
    
    clusterA = list()
    clusterB = list()
    
    similarityMatrix = similarity(matrix, centroids)
    
    for index in range(similarityMatrix.shape[0]):
        similarityRow = similarityMatrix[index]
        
        #Sort the index of the matrix in ascending order of value and get the index of the last element
        #This index will be the cluster that the row in input matrix will belong to
        similaritySorted = np.argsort(similarityRow)[-1]
        
        if similaritySorted == 0:
            clusterA.append(index)
        else:
            clusterB.append(index)
        
    return clusterA, clusterB


# In[8]:


def recalculateCentroid(matrix, clusters):
    centroids = list()
    
    for i in range(0,2):
        cluster = matrix[clusters[i],:]
        clusterMean = cluster.mean(0)
        centroids.append(clusterMean)
        
    centroids_array = np.asarray(centroids)
    
    return centroids_array


# In[9]:


def kmeans(matrix, numberOfIterations):
    
    centroids = initialCentroids(matrix)
    
    for _ in range(numberOfIterations):
        
        clusters = list()
        
        clusterA, clusterB = findClusters(matrix, centroids)
        
        if len(clusterA) > 1:
            clusters.append(clusterA)
        if len(clusterB) > 1:
            clusters.append(clusterB)
            
        centroids = recalculateCentroid(matrix, clusters)
        
    return clusterA, clusterB


# In[10]:


def calculateSSE(matrix, clusters):
    
    SSE_list = list()
    SSE_array = []
    
    for cluster in clusters:
        members = matrix[cluster,:]
        SSE = np.sum(np.square(members - np.mean(members)))
        SSE_list.append(SSE)
        
    SSE_array = np.asarray(SSE_list)
    dropClusterIndex = np.argsort(SSE_array)[-1]
            
    return dropClusterIndex


# In[11]:


def bisecting_kmeans(matrix, k, numberOfIterations):
    
    clusters = list()
    
    initialcluster = list()
    for i in range(matrix.shape[0]):
        initialcluster.append(i)
    
    clusters.append(initialcluster)
    
    while len(clusters) < k:

        dropClusterIndex = calculateSSE(matrix, clusters)
        droppedCluster = clusters[dropClusterIndex]
        
        clusterA, clusterB = kmeans(matrix[droppedCluster,:], numberOfIterations)
        del clusters[dropClusterIndex]
        
        actualClusterA = list()
        actualClusterB = list()
        for index in clusterA:
            actualClusterA.append(droppedCluster[index])
            
        for index in clusterB:
            actualClusterB.append(droppedCluster[index])
        
        clusters.append(actualClusterA)
        clusters.append(actualClusterB)
    
    labels = [0] * matrix.shape[0]

    for index, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = index + 1
    return labels


# In[12]:


#Read CSR matrix from the input file
csrMatrix = csr_read('train.dat', ftype="csr", nidx=1)

#Scale the CSR matrix by idf (Inverse Document Frequency)
csrIDF = csr_idf(csrMatrix, copy=True)

#Normalize the rows of a CSR matrix by their L-2 norm.
csrL2Normalized = csr_l2normalize(csrIDF, copy=True)

#Obtain a dense ndarray representation of the CSR matrix.
denseMatrix = csrL2Normalized.toarray()
print("Dense:",denseMatrix)

# In[13]:

labels = bisecting_kmeans(denseMatrix, 7, 10)
# write result to output file
    outputFile = open("output.dat", "w")
    for index in labels:
        outputFile.write(str(index) +'\n')
    outputFile.close()
kValues = list()
scores = list()

# for k in range(3, 22, 2):
#     labels = bisecting_kmeans(denseMatrix, k, 10)
#     print("Iterated",k)    
#     score = calinski_harabaz_score(denseMatrix, labels)
#     kValues.append(k)
#     scores.append(score)
#     print ("For K= %d Calinski Harabaz Score is %f" %(k, score))


# # In[ ]:


# #get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt

# plt.plot(kValues, scores)
# plt.xticks(kValues, kValues)
# plt.xlabel('Number of Clusters k')
# plt.ylabel('Calinski and Harabaz Score')
# plt.title('Trend of Average Distance to Centroid/Diameter')
# plt.grid(linestyle='dotted')
# plt.savefig('plot.png')
# plt.show()
