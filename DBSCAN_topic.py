from sklearn.cluster import DBSCAN
import numpy as np
import PCA_plot3D as pca
import operator


def DBSCAN_Topic(word_vect_dict):
    X = []
    for index in range(0, len(word_vect_dict)):
        X.append(list(word_vect_dict.values())[index])
    bestCluster = {}
    best_eps = {}
    for i in range(2, 10):
        clustering = DBSCAN(eps=i, min_samples=2).fit(X)

        key = []
        value = []
        d = {}
        for index in range(0, len(word_vect_dict)):
            d[clustering.labels_[index]] = 0

        for index in range(0, len(word_vect_dict)):
            d[clustering.labels_[index]] = d[clustering.labels_[index]] + 1

        cluster_array_sorted = sorted(d.items(), key=operator.itemgetter(1),
                                      reverse=True)  # clusters ordinati in base al numero di elementi
        number_of_clusters = len(cluster_array_sorted)  # abbiamo trovato il numero di cluster diversi

        # if len(cluster_array_sorted) > 0:
        #     if cluster_array_sorted[0][0] != -1:
        #         best_eps[i] = number_of_clusters # numero di cluster diversi
        #     else:

        if cluster_array_sorted[0][0] != -1:
            cluster_array_sorted = cluster_array_sorted[0][0]
        elif len(cluster_array_sorted) == 1:
            continue
        else:
            cluster_array_sorted = cluster_array_sorted[1][0]

        bestCluster[i] = cluster_array_sorted
        best_eps[i] = number_of_clusters  # id raggio piu popoloso

        for index in range(0, len(word_vect_dict)):
            key.append(clustering.labels_[index]) # prendo gli id dei cluster
            value.append(list(word_vect_dict.values())[index])

       # pca.pca_clustering_3D(value, key)

    theBest = sorted(best_eps.items(), key=operator.itemgetter(1), reverse=True)
    clustering = DBSCAN(eps=theBest[0][0], min_samples=2).fit(X) # clustering sul raggio che ha il maggior numero di cluster
    # for index in range(0, len(word_vect_dict)):
    #     if (c == clustering.labels_[index]):
    #         key.append(clustering.labels_[index])
    #         value.append(list(word_vect_dict.values())[index])


    for index in range(0, len(word_vect_dict)):
        key.append(clustering.labels_[index])
        value.append(list(word_vect_dict.values())[index])

   # pca.pca_clustering_3D(value, key)

    key = []
    value = []
    word = []
    for index in range(0, len(word_vect_dict)):
        if (bestCluster[theBest[0][0]] == clustering.labels_[index]):
            key.append(clustering.labels_[index])
            value.append(list(word_vect_dict.values())[index])
            word.append(list(word_vect_dict.keys())[index])

    return word, value
