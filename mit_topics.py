import csv
import logging
import coloredlogs
import os
import threading
from datetime import datetime
import time
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
import text_preprocessing as tp
import embedding_word as ew
import centroid_topic as ct
import CSV_complie as cc
import PCA_plot3D as pca
import DBSCAN_topic as db
from tqdm import tqdm
import top_2_vec as t2v
import multiprocessing
from csv import writer
import pandas as pd
import operator

#GLOBAL VARIABLES
choose = ""


'''
    multiprocess function to preprocess the text
'''

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                    fmt="- Process -> pid[%(process)d], name[%(processName)s] Function -> [%(funcName)s]\n%(asctime)s --- %(levelname)s log -> [%(message)s]")


def parallelized_function(file):
    if file.endswith(".txt"):
        input_file = open(f"data/{file}", encoding="utf8")
        file_text = input_file.read()
        file_text = tp.remove_whitespace(file_text)  # rimozione doppi spazi
        file_text = tp.tokenization(file_text)  # tokenizzo
        file_text = tp.stopword_removing(file_text)  # rimuovo le stopword
        file_text = tp.pos_tagging(file_text)  # metto un tag ad ogni parola
        file_text = tp.lemmatization(file_text)  # trasformo nella forma base ogni parola

        return file_text

def choice_b(tot_vectors):
    word_vector, value_vactor = db.DBSCAN_Topic(tot_vectors)
    # value_vactor =  list(tot_vectors.values())
    # word_vector = list(tot_vectors.keys())
    tot_vectors = {}
    for i in range(0, len(word_vector)):
        tot_vectors[word_vector[i]] = value_vactor[i]

    # rimuovo gli outlier e creo il file
    transformer = RobustScaler(quantile_range=(25.0, 75.0)).fit(value_vactor)
    #  pca.pca_clustering_3D(transformer.transform(value_vactor), list(tot_vectors.keys()), f"/html/{file[: -4]}")

    sortedDist = ct.centroid_Topic(transformer.transform(value_vactor), word_vector)
    # print(sortedDist)
    return word_vector

def choice_d(tot_vectors,file_text):
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
                words.append(sortedDist[j][0])

    tp.tag_cloud(words)

def choice_e(list_files):
    documents = []
    for file in list_files:
        if file.endswith(".txt"):
            input_file = open(f"data/{file}", encoding="utf8")
            file_text = input_file.read()
            documents.append(file_text)
    t2v.top_2_vec(documents)

if __name__ == "__main__":

    while 1:
        choose = input('Insert:\n'
                       'a) If you want the cluster centroid\n'
                       'b) If you want the centroid of the densest area of the cluster\n'
                       'c) If you want to see the most frequent words of the cluster\n'
                       'd) If you want to see the most frequent words of the densest part of the cluster\n'
                       'e) If you want use to2vec to detect macro topics on all documents\n')

        if choose == "a":
            break
        elif choose == "b":
            break
        elif choose == "c":
            break
        elif choose == "d":
            break
        elif choose == "e":
            break

    # Working Folder
    os.chdir("data")
    listDoc = os.listdir()
    os.chdir("../")


    decade = input("Insert the decade: \n(insert skip if you want to scan all the documents)\n")

    # take all the names of the files
    filtered_docs_list = []
    all_docs = []

    for doc in listDoc:
        if doc.endswith(".txt") and decade in doc:
            filtered_docs_list.append(doc)

    if filtered_docs_list == []:
        print("No documents found for this decade")
        exit()

    if(choose != "e"):
        # preprocess data

        print("You have ", multiprocessing.cpu_count(), " cores")
        core_number = input('How many subprocess do you want to use?: (Do not overdo it)\n')

        logger.info("Start Time : %s", datetime.now())
        start_time = datetime.utcnow()

        pool = multiprocessing.Pool(processes=int(core_number))

        # array of documents preprocessed
        results = [pool.map(parallelized_function, filtered_docs_list)]

        # logs
        pool.close()
        logger.info("End Time : %s", datetime.now())
        end_time = datetime.utcnow()
        total_time = end_time - start_time
        logger.info("Total Time : %s", total_time)

        # concat all the documents
        concat_results = np.concatenate(results[0])
        concat_results_copy = concat_results

        #tag cloud of most frequent words of the decade
        if (choose == "c"):
            tp.tag_cloud(concat_results)

        #remove duplicates
        concat_results = [list(dict.fromkeys(concat_results))]

        tot_vectors = {}
        for word in concat_results[0]:
            tot_vectors[str(word)] = ew.get_embedding(str(word))

        # tag cloud of most frequent words of the densest part of the decade
        if (choose == "d"):
            choice_d(tot_vectors, concat_results_copy)

        if(choose == "b"):
            topWords = choice_b(tot_vectors)[:30]
            print(topWords)

            with open(f'output/{decade}_30TopWords.csv', 'w') as f:
                mywriter = csv.writer(f, delimiter='\n')
                mywriter.writerows(topWords)
    else:
        # use to2vec to detect topics of decade
        choice_e(filtered_docs_list)