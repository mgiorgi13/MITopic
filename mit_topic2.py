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
import sys


# GLOBAL VARIABLES
choose = ""

'''
    multiprocess function to preprocess the text
'''
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                    fmt="- Process -> pid[%(process)d], name[%(processName)s] Function -> [%(funcName)s]\n%(asctime)s --- %(levelname)s log -> [%(message)s]")


def preprocessing(file):
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
def frequencyEachDoc(files, filtered_docs_list, year):
    frequency_list = []
    header = ['file_name', 'word_frequency']

    with open(f'output/{year}_file_word_frequency.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

    for i in range(len(filtered_docs_list)):
        with open(f'output/{year}_file_word_frequency.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write file words
            frequency_list.append(tp.word_count(files[i]))
            data = [filtered_docs_list[i],
                    str(frequency_list[i]).replace(",", "").replace("[", "").replace("]", "")]
            writer.writerow(data)

def printCloud(files):
    for file in files:
        tp.tag_cloud(file)

def densityArea(docs,title,year):
    for i in range(0, len(docs)):
        clear_results = [list(dict.fromkeys(docs[i]))]
        tot_vectors = {}
        for word in clear_results[0]:
            tot_vectors[str(word)] = ew.get_embedding(str(word))
        if os.path.exists(f"html/{year}/{title[i][:-4]}") == 0:
            os.makedirs(f"html/{year}/{title[i][:-4]}")
        pca.pca_clustering_3D(list(tot_vectors.values()), list(tot_vectors.keys()),
                              f"/html/{year}/{title[i][:-4]}/InitialCluster__nWords_{len(tot_vectors)}")
        word_vector, value_vactor, radius = db.DBSCAN_Topic(tot_vectors, year, title[i][:-4])
        # value_vactor =  list(tot_vectors.values())
        # word_vector = list(tot_vectors.keys())
        tot_vectors = {}
        for j in range(0, len(word_vector)):
            tot_vectors[word_vector[j]] = value_vactor[j]

        # rimuovo gli outlier e creo il file
        transformer = RobustScaler(quantile_range=(25.0, 75.0)).fit(value_vactor)
        pca.pca_clustering_3D(transformer.transform(value_vactor), list(tot_vectors.keys()),
                              f"/html/{year}/{title[i][:-4]}/FinalCluster__radiusOfDensisty_{radius}__year_{year}__nWords_{len(value_vactor)}")

        sortedDist = ct.centroid_Topic(transformer.transform(value_vactor), word_vector)
        # print(sortedDist)
        word_vector = []
        for j in range(0, len(sortedDist)):
            word_vector.append(sortedDist[j][0])
        sim = []
        unsim = []
        zer = []
        for j in range(0, len(sortedDist)):
            if sortedDist[j][1] > 0:
                sim.append(sortedDist[j][0])
            if sortedDist[j][1] < 0:
                unsim.append(sortedDist[j][0])
            if sortedDist[j][1] == 0:
                zer.append(sortedDist[j][0])
        topWords = sim[:100]
        path = f"output/{year}/{title[i][:-4]}"
        if os.path.exists(path) == 0:
            os.makedirs(path)
        with open(f'{path}/{year}_TopWords.csv', 'w', encoding='UTF8') as f:
            mywriter = csv.writer(f, delimiter='\n')
            mywriter.writerows([topWords])

if __name__ == "__main__":

    # print(sys.argv[0])  # prints python_script.py
    # print(sys.argv[1])  # prints var1 choose option
    # print(sys.argv[2])  # prints var2 year
    # print(sys.argv[3])  # prints var3 num cores

    if (len(sys.argv) == 4):
        arg_from_command_line = True
    else:
        arg_from_command_line = False


    if (arg_from_command_line == False):
        year = input("Insert year to be analyze: \n(insert skip if you want to scan all the documents)\n")
    else:
        year = str(sys.argv[2])

    # Working Folder
    os.chdir("data")
    listDoc = os.listdir()
    os.chdir("../")

    filtered_docs_list = []
    all_docs = []

    for doc in listDoc:
        if doc.endswith(".txt"):
            all_docs.append(doc)
        if doc.endswith(".txt") and year in doc:
            filtered_docs_list.append(doc)

    if filtered_docs_list == [] and year != "skip":
        print("No documents found for this decade")
        exit()

    if year == "skip":
        filtered_docs_list = all_docs

    # if (arg_from_command_line == False):
    #     print("You have ", multiprocessing.cpu_count(), " cores")
    #     core_number = input('How many core do you want to use?: (Do not overdo it)\n')
    # else:
    #     core_number = str(sys.argv[3])

    logger.info("Start Time : %s", datetime.now())
    start_time = datetime.utcnow()
    filePP = []
    for doc in filtered_docs_list:
        filePP.append(preprocessing(doc))
    a = 2
    frequencyEachDoc(filePP, filtered_docs_list, year) #write in file "output" all frequency words each document for year
  # printCloud(filePP) #plot of cloud using the cloud
    densityArea(filePP,filtered_docs_list,year) #found the densest area of the cluster