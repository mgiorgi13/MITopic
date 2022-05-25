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

# GLOBAL VARIABLES
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

        logger.info("Subprocess for file -> [%s]", file)

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
                words.append(sortedDist[j][0])

    tp.tag_cloud(words)


def choice_e(list_files):
    topic_words, word_scores, topic_nums = t2v.top_2_vec(list_files)
    return topic_words, word_scores, topic_nums


def printToFile(topicResults):
    with open('output/results.csv', 'w') as csvfile:
        fieldnames = ['File', 'TopicWords', 'WordScore', 'TopicNumber']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, len(topicResults)):
            writer.writerow(
                {'File': topicResults[i][0], 'TopicWords': topicResults[i][1], 'WordScore': topicResults[i][2],
                 'TopicNumber': topicResults[i][3]})


if __name__ == "__main__":

    while 1:
        choose = input('Insert:\n'
                       'b) If you want the centroid of the densest area of the cluster\n'
                       'c) If you want to see the most frequent words of the cluster\n'
                       'd) If you want to see the most frequent words of the densest part of the cluster\n'
                       'e) If you want use top2vec to detect macro topics on all documents\n')

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

    if choose != "e":
        # preprocess data

        year = input("Insert year to be analyze: \n(insert skip if you want to scan all the documents)\n")

        # take all the names of the files
        filtered_docs_list = []
        all_docs = []

        for doc in listDoc:
            if doc.endswith(".txt"):
                input_file = open(f"data/{doc}", encoding="utf8")
                file_text = input_file.read()
                all_docs.append(file_text)
            if doc.endswith(".txt") and year in doc:
                filtered_docs_list.append(doc)

        if filtered_docs_list == [] and year != "skip":
            print("No documents found for this decade")
            exit()

        print("You have ", multiprocessing.cpu_count(), " cores")
        core_number = input('How many core do you want to use?: (Do not overdo it)\n')

        logger.info("Start Time : %s", datetime.now())
        start_time = datetime.utcnow()

        pool = multiprocessing.Pool(processes=int(core_number))  # creation of the pool of processes

        results = [pool.map(parallelized_function, filtered_docs_list)]  # array of documents preprocessed

        pool.close()  # close the pool of processes

        # logs the time of the process
        logger.info("End Time : %s", datetime.now())
        end_time = datetime.utcnow()
        total_time = end_time - start_time
        logger.info("Total Time : %s", total_time)

        concat_results = np.concatenate(results[0])  # concat all the processed documents
        # concat_results_copy = concat_results

        # tag cloud of most frequent words of the decade
        if choose == "c":
            tp.tag_cloud(concat_results)
        elif choose == "d":
            clear_results = [list(dict.fromkeys(concat_results))]  # remove duplicates
            tot_vectors = {}
            for word in clear_results[0]:
                tot_vectors[str(word)] = ew.get_embedding(str(word))
            choice_d(tot_vectors,
                     clear_results[0])  # tag cloud of most frequent words of the densest part of the decade
        elif choose == "b":
            clear_results = [list(dict.fromkeys(concat_results))]  # remove duplicates
            tot_vectors = {}
            for word in clear_results[0]:
                tot_vectors[str(word)] = ew.get_embedding(str(word))
            topWords = choice_b(tot_vectors)[:30]  # get the centroid of the densest area of the cluster
            # print(topWords)

            # print results of the centroid of the densest area of the cluster in file
            with open(f'output/{year}_30TopWords.csv', 'w') as f:
                mywriter = csv.writer(f, delimiter='\n')
                mywriter.writerows([topWords])
    else:
        # execute top_2_vec on documents grouped by five years
        year_list = []
        for doc in listDoc:
            year = doc.split("_")[2]
            if year not in year_list:
                year_list.append(year)

        year_list.sort()

        # extract interval of 10 year from the list of years
        year_list_10 = []
        for i in range(0, len(year_list) - 9, 10):
            year_list_10.append(year_list[i:i + 10])
        year_list_10.append(year_list[len(year_list) - len(year_list) % 10:])  # take the remaining years

        resultsForFile = []

        for group in year_list_10:
            list_files = []
            for year in group:
                for doc in listDoc:
                    if doc.endswith(".txt") and year in doc:
                        input_file = open(f"data/{doc}", encoding="utf8")
                        file_text = input_file.read()
                        list_files.append(file_text)
            logger.info("Start Top2Vec analysis for documents conteined in the year: %s. Number of documents: %s",
                        group, len(list_files))
            topic_words, word_scores, topic_nums = choice_e(list_files)
            partial_results = [group, topic_words, word_scores, topic_nums]
            logger.info("End Top2Vec analysis for documents conteined in the year: %s", group)
            resultsForFile.append(partial_results)

        printToFile(resultsForFile)
