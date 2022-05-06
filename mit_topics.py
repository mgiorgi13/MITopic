import os
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
import text_preprocessing as tp
import embedding_word as ew
import centroid_topic as ct
import  PCA_plot3D as pca
import  DBSCAN_topic as db
from tqdm import tqdm

import pandas as pd
import operator


if __name__ == "__main__":
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

            file_text = tp.remove_whitespace(file_text) #rimozione doppi spazi
            file_text = tp.tokenization(file_text)  #tokenizzo
            file_text = tp.stopword_removing(file_text)   #rimuovo le stopword
            file_text = tp.pos_tagging(file_text) #metto un tag ad ogni parola
            file_text = tp.lemmatization(file_text) #trasformo nella forma base ogni parola
            tot_vectors = {}
            for word in tqdm(file_text):
                tot_vectors[word] = ew.get_embedding(word)

            db.DBSCAN_Topic(tot_vectors)
            value_vactor =  list(tot_vectors.values())
            word_vector = list(tot_vectors.keys())

            pca.pca_clustering_3D( value_vactor, word_vector)

            # rimuovo gli outlier e creo il file
            transformer = RobustScaler(quantile_range=(25.0, 75.0)).fit(value_vactor)
            pca.pca_clustering_3D(transformer.transform(value_vactor), list(tot_vectors.keys()),f"/html/{file[: -4]}")


            sortedDist =  ct.centroid_Topic(transformer.transform(value_vactor),word_vector)


            #prendere ogni vettore, togliere outlier, calolo centro(uno per file), prendere parola piú vicina al centro(vettore) usando la distanza, recuperare la parola,
            # calcolo la distanza nella dimensione orginale
            #tp.tag_cloud(file_text) #stamo in base alla frequeza di ogni parola
            break