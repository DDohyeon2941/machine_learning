# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csgraph
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets
from sklearn.manifold import MDS, Isomap
import matplotlib.pyplot as plt


def get_knn_graph(X,k):
    '''
    parameters
    ----------
    X : 2-D array
     input data matrix
    k : int
     the number of nearest neighbors
    
    Notes
    ----------
    knn graph whose element ij is distance between xi and xj if xj is in knn of xi
    
    return
    ----------
    knn : csr_matrix(shape = len(X) * len(X))
     pairwise distance matrix of samples
    
    '''
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    
    return neigh.kneighbors_graph(mode='distance')
    
def cal_dijkstra(knn):
    '''
    parameters
    ----------
    knn : 2-D array
        knn graph
    
    Notes
    ----------
    make distance_matrix based on knn by Dijkstra algorithm
    
    
    return
    ----------
    dist_matrix : 2-D array
     pairwise distance matrix among samples
    '''
    dist_matrix = csgraph.dijkstra(csgraph=knn, directed=False)
    
    return dist_matrix

def isomap(X,k,n_dimension):
    '''
    parameters
    ----------
    X : 2-D array
     input data matrix
    k : int
     the number of nearest neighbors
    n_dimension: int
     reduced dimensionality
    
    Notes
    ----------
     to get result for return use get_knn_graph, cal_dijkstra
     objective function of MDS, ISO is same => distance matrix can be used as input of MDS
    
    return 
    ----------
    2-D array
     low dimensional coordinates
    '''
    
    knn=get_knn_graph(X,k)
    dist_matrix=cal_dijkstra(knn)
    
    embedding = MDS(n_components=n_dimension, dissimilarity='precomputed')

    X_transformed=embedding.fit_transform(dist_matrix)

    return X_transformed
 #%%   
if __name__== "__main__":
    n_points = 1000
    X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
    
    k=5
    n_dimension=2
    # TODO: get low dimensional coordinates using sklearn
    X_low=isomap(X,k,n_dimension)
    
    # TODO: Compare
    ##get result using Isomap
    embedding1=Isomap(n_neighbors=k,n_components=n_dimension, path_method='D')
    X_iso=embedding1.fit_transform(X)
    
    ##plot_and_compare
    plt.scatter(X_iso[:,0], X_iso[:,1], c=color)
    plt.scatter(X_low[:,0], X_low[:,1], c=color)
