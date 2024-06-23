from sklearn.cluster import DBSCAN
import numpy as np

def cluster(points,cluster_treshold,clustering_cluster_number_trashold,cluster_centers_prev):
    if np.size(points) > 0:
        #ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        #ms = MeanShift(bandwidth=bandwidth,  GPU = True)
        ms = DBSCAN(eps =50, min_samples = 1)
        ms.fit(points)
        labels = ms.labels_
        
        #cluster_centers = ms.cluster_centers_
        cluster_centers = np.array([np.mean(points[ms.labels_ == v],axis = 0) for v in np.unique(ms.labels_)])

        if np.size(cluster_centers_prev):
            for i,cluster_center in enumerate(cluster_centers):
                if np.min(np.abs(np.sum(cluster_center - cluster_centers_prev,axis = 1)))<cluster_treshold:
                    np.delete(cluster_centers,i,0)
                    
        cluster_centers_prev = cluster_centers
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        flag_clustering = n_clusters_ > clustering_cluster_number_trashold
        return cluster_centers, cluster_centers_prev, n_clusters_, flag_clustering



def cluster_naive(points,cluster_treshold,clustering_cluster_number_trashold,cluster_centers_prev):
    distance_trashold = 70
    point_number = 0
    points = np.array(points)
    while point_number < len(points):
        tmp_point = points[point_number,:]
        points = points[(np.abs(points - points[point_number,:])>=distance_trashold).all(axis = 1)]
        points = np.r_[[tmp_point],points]
        point_number += 1
    return points, points, len(points), False
        
