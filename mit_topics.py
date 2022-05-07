import os
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
import text_preprocessing as tp
import embedding_word as ew
import centroid_topic as ct
import PCA_plot3D as pca
import DBSCAN_topic as db
from tqdm import tqdm

import pandas as pd
import operator


def choice_a(tot_vectors):
    value_vactor = list(tot_vectors.values())
    word_vector = list(tot_vectors.keys())
    # rimuovo gli outlier e creo il file
    transformer = RobustScaler(quantile_range=(25.0, 75.0)).fit(value_vactor)
    pca.pca_clustering_3D(transformer.transform(value_vactor), list(tot_vectors.keys()), f"/html/{file[: -4]}")

    sortedDist = ct.centroid_Topic(transformer.transform(value_vactor), word_vector)
    print(sortedDist)


def choice_b(tot_vectors):
    word_vector, value_vactor = db.DBSCAN_Topic(tot_vectors)
    # value_vactor =  list(tot_vectors.values())
    # word_vector = list(tot_vectors.keys())
    tot_vectors = {}
    for i in range(0, len(word_vector)):
        tot_vectors[word_vector[i]] = value_vactor[i]

    # rimuovo gli outlier e creo il file
    transformer = RobustScaler(quantile_range=(25.0, 75.0)).fit(value_vactor)
    pca.pca_clustering_3D(transformer.transform(value_vactor), list(tot_vectors.keys()), f"/html/{file[: -4]}")

    sortedDist = ct.centroid_Topic(transformer.transform(value_vactor), word_vector)
    print(sortedDist)
    return word_vector


def choice_c(file_text):
    tp.tag_cloud(file_text)


def choice_d(tot_vectors, file_text):
    word_vector, value_vactor = db.DBSCAN_Topic(tot_vectors)
    tot_vectors = {}
    for i in range(0, len(word_vector)):
        tot_vectors[word_vector[i]] = value_vactor[i]
    # rimuovo gli outlier e creo il file
    transformer = RobustScaler(quantile_range=(25.0, 75.0)).fit(value_vactor)

    sortedDist = ct.centroid_Topic(transformer.transform(value_vactor), word_vector)
    words = []
    for i in range(0, len(file_text)):
        for j in range(0, len(sortedDist)):
            if sortedDist[j][0] == file_text[i]:
                words.append(sortedDist[j])

    tp.tag_cloud(words)


if __name__ == "__main__":
    while (1):
        choose = input('Insert:\n'
                       'a) If you want the cluster centroid\n'
                       'b) If you want the centroid of the densest area of the cluster\n'
                       'c) If you want to see the most frequent words of the cluster\n'
                       'd) If you want to see the most frequent words of the densest part of the cluster\n')
        if choose == "a":
            break
        elif choose == "b":
            break
        elif choose == "c":
            break
        elif choose == "d":
            break

    # Working Folder
    os.chdir("data")
    listDoc = os.listdir()
    os.chdir("../")
    count = 0
    for file in listDoc:
        count = count + 1

        if file.endswith(".txt"):
            input_file = open(f"data/{file}", encoding="utf8")
            file_text = input_file.read()

            file_text = tp.remove_whitespace(file_text)  # rimozione doppi spazi
            file_text = tp.tokenization(file_text)  # tokenizzo
            file_text = tp.stopword_removing(file_text)  # rimuovo le stopword
            file_text = tp.pos_tagging(file_text)  # metto un tag ad ogni parola
            file_text = tp.lemmatization(file_text)  # trasformo nella forma base ogni parola
            tot_vectors = {}
            for word in tqdm(file_text):
                tot_vectors[word] = ew.get_embedding(word)

            if choose == "a":
                choice_a(tot_vectors)
            elif choose == "b":
                choice_b(tot_vectors)
            elif choose == "c":
                choice_c(file_text)
            else:
                choice_d(tot_vectors, file_text)
            break
