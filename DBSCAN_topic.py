import os

from sklearn.cluster import DBSCAN
import numpy as np
import PCA_plot3D as pca
import operator


def DBSCAN_Topic(word_vect_dict, year, title):
    print("partito dbscan")
    X = []
    for index in range(0, len(word_vect_dict)):
        X.append(list(word_vect_dict.values())[index])
    bestCluster = {}
    best_eps = {}
    for i in range(1, 11):
        clustering = DBSCAN(metric = 'cosine', eps=i/10, min_samples=1).fit(X)

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
        # find how many elements are in each cluster
        d = {}
        for index in range(0, len(key)):
            d[key[index]] = 0
        for index in range(0, len(key)):
            d[key[index]] = d[key[index]] + 1

        pca.pca_clustering_3D(value, key, f"//html/{year}/{title}/year_{year}__radius_{i}")

    theBest = sorted(best_eps.items(), key=operator.itemgetter(1), reverse=True)
    clustering = DBSCAN(eps=theBest[0][0]/10, min_samples=3).fit(
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

    key = []
    value = []
    word = []
    for index in range(0, len(word_vect_dict)):
        if (bestCluster[theBest[0][0]] == clustering.labels_[index]):
            key.append(clustering.labels_[index])
            value.append(list(word_vect_dict.values())[index])
            word.append(list(word_vect_dict.keys())[index])

    return word, value, theBest[0][0]

from sklearn.cluster import DBSCAN
import numpy as np
import PCA_plot3D as pca
import operator


def DBSCAN_Topic2(word_vect_dict, year, min_simple, loop, dir):
    min_simple= 2
    print("partito dbscan")
    X = []
    for index in range(0, len(word_vect_dict)):
        X.append(list(word_vect_dict.values())[index])
    bestCluster = {}
    best_eps = {}
    for i in range(1, 11):
        clustering = DBSCAN(metric = 'cosine' ,eps=i/10, min_samples=min_simple).fit(X)

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
        path = f"html/{year}/{dir}"
        if os.path.exists(path) == False:
            os.makedirs(path)
        pca.pca_clustering_3D(value, key, f"/{path}/year_{year}_radius{i}")

    theBest = sorted(best_eps.items(), key=operator.itemgetter(1), reverse=True)
    clustering = DBSCAN(metric = 'cosine',eps=theBest[0][0]/10, min_samples=min_simple).fit(
        X)  # clustering sul raggio che ha il maggior numero di cluster

    # for index in range(0, len(word_vect_dict)):
    #     if (c == clustering.labels_[index]):
    #         key.append(clustering.labels_[index])
    #         value.append(list(word_vect_dict.values())[index])

    key = []
    value = []
    word = []
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
    lenHigt = 0
    clstr = 0
    if loop == 0:
        #vedo quale cluster è più grande e lo prendo
        for t in range(0, len(dctWord)):
            if(len(dctWord[t])> lenHigt):
                lenHigt = len(dctWord[t])
                clstr = t
        #creo dizionario
        word_vect_dict = {}
        for t in range(0, lenHigt):
            word_vect_dict[dctWord[clstr][t]] = dctValue[clstr][t]

        clustering= DBSCAN_Topic2(word_vect_dict,year,3,1, "clusterOfCluster")

    for index in range(0, len(word_vect_dict)):
        if (clustering.labels_[index] != -1):
            key.append(clustering.labels_[index])
            value.append(list(word_vect_dict.values())[index])
            word.append(list(word_vect_dict.keys())[index])
    # key = []
    # value = []
    # word = []
    # for index in range(0, len(word_vect_dict)):
    #     if (bestCluster[theBest[0][0]] == clustering.labels_[index]):
    #         key.append(clustering.labels_[index])
    #         value.append(list(word_vect_dict.values())[index])
    #         word.append(list(word_vect_dict.keys())[index])

    return word, value, theBest[0][0]