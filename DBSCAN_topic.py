from sklearn.cluster import DBSCAN
import numpy as np
import PCA_plot3D as pca
import operator


def DBSCAN_Topic(word_vect_dict, year):
    print("partito dbscan")
    X = []
    for index in range(0, len(word_vect_dict)):
        X.append(list(word_vect_dict.values())[index])
    bestCluster = {}
    best_eps = {}
    for i in range(1, 11):
        clustering = DBSCAN(metric='cosine', eps=i / 10, min_samples=5).fit(X)

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
            key.append(clustering.labels_[index])  # prendo gli id dei cluster
            value.append(list(word_vect_dict.values())[index])

 #       pca.pca_clustering_3D(value, key, f"/html/{year}_gtd/year_{year}__radius_{i / 10}_FinalClustering")

    theBest = sorted(best_eps.items(), key=operator.itemgetter(1), reverse=True)
    clustering = DBSCAN(metric='cosine', eps=theBest[0][0] / 10, min_samples=5).fit(
        X)  # clustering sul raggio che ha il maggior numero di cluster

    dctWord = {}
    dctValue = {}

    for index in range(0, len(word_vect_dict)):
        if (clustering.labels_[index] != -1):
            dctWord[clustering.labels_[index]] = []
            dctValue[clustering.labels_[index]] = []

    for index in range(0, len(word_vect_dict)):
        if (clustering.labels_[index] != -1):
            dctWord[clustering.labels_[index]].append(list(word_vect_dict.keys())[index])
            dctValue[clustering.labels_[index]].append(list(word_vect_dict.values())[index])

    # for k in sorted(dctWord, key=lambda k: len(dctWord[k]), reverse=True):
    #     print(k, len(dctWord[k]))
    #     print(dctWord[k][:50])
    
    with open(f"output/{year}.txt", "w") as f:
        f.write("selected year: " + year)
        f.write(" \n")
        for k in sorted(dctWord, key=lambda k: len(dctWord[k]), reverse=True):

            f.write("len: " + str(len(dctWord[k])))
            f.write(" \n")
            f.write("cluster words:\n")
            for word in dctWord[k][:50]:
                f.write(word + ", ")

            f.write(" \n")
            f.write(" \n")

    key = []
    value = []
    word = []
    for index in range(0, len(word_vect_dict)):
        if (bestCluster[theBest[0][0]] == clustering.labels_[index]):
            key.append(clustering.labels_[index])
            value.append(list(word_vect_dict.values())[index])
            word.append(list(word_vect_dict.keys())[index])
    pca.pca_clustering_3D(value, key, f"/html/{year}_gtd/year_{year}__radius_{theBest[0][0]/10}_FinalClustering")
    return word, value, theBest[0][0]
