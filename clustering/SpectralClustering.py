from sklearn.cluster import SpectralClustering

def spectral_clustering(affinity_matrix, n_clusters):
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    return clustering.fit_predict(affinity_matrix)