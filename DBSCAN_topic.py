from sklearn.cluster import DBSCAN
import numpy as np
import PCA_plot3D as pca
import operator


def DBSCAN_Topic(word_vect_dict):
    X = []
    for index in range(0, len(word_vect_dict)):
        X.append(list(word_vect_dict.values())[index])

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

        c = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        lunc = len(c)
        if c[0][0] != -1:
            c = c[0][0]
        elif len(c) == 1:
            return
        else:
            c = c[1][0]

        best_eps[i] = lunc

    theBest = sorted(best_eps.items(), key=operator.itemgetter(1), reverse=True)
    clustering = DBSCAN(eps=theBest[0][0], min_samples=2).fit(X)
    # for index in range(0, len(word_vect_dict)):
    #     if (c == clustering.labels_[index]):
    #         key.append(clustering.labels_[index])
    #         value.append(list(word_vect_dict.values())[index])

    word = []
    for index in range(0, len(word_vect_dict)):
        if (c == clustering.labels_[index]):
            key.append(clustering.labels_[index])
            value.append(list(word_vect_dict.values())[index])
            word.append(list(word_vect_dict.keys())[index])

    #    pca.pca_clustering_3D(value, key)
    return word, value
