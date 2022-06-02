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
import lda as lda
import lsa as lsa

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


def choice_b(tot_vectors, year):
    pca.pca_clustering_3D(list(tot_vectors.values()), list(tot_vectors.keys()),
                          f"/html/InitialCluster__year_{year}__nWords_{len(tot_vectors)}")
    word_vector, value_vactor, radius = db.DBSCAN_Topic(tot_vectors, year)
    # value_vactor =  list(tot_vectors.values())
    # word_vector = list(tot_vectors.keys())
    tot_vectors = {}
    for i in range(0, len(word_vector)):
        tot_vectors[word_vector[i]] = value_vactor[i]

    # rimuovo gli outlier e creo il file
    # transformer = RobustScaler(quantile_range=(25.0, 75.0)).fit(value_vactor)
    transformer = RobustScaler(quantile_range=(0, 75.0))
    transformer.fit(list(tot_vectors.values()))
    centroid_ = transformer.center_
    centroid_ = np.array([centroid_])
    distance_vector = {}
    for j in range(0, len(tot_vectors) - 1):
        dist = cosine_similarity(centroid_, np.array([list(tot_vectors.values())[j]]))
        distance_vector[list(tot_vectors.keys())[j]] = dist[0][0]
    distance_vector = sorted(distance_vector.items(), key=operator.itemgetter(1), reverse=True)

    # pca.pca_clustering_3D(transformer.transform(value_vactor), list(tot_vectors.keys()),
    #                       f"/html/FinalCluster__radiusOfDensisty_{radius}__year_{year}__nWords_{len(value_vactor)}")
    pca.pca_clustering_3D(value_vactor, list(tot_vectors.keys()),
                          f"/html/FinalCluster__radiusOfDensisty_{radius}__year_{year}__nWords_{len(value_vactor)}")

    # sortedDist = ct.centroid_Topic(transformer.transform(value_vactor), word_vector)
    #
    # word_vector = []
    # for i in range(0, len(sortedDist)):
    #     word_vector.append(sortedDist[i][0])
    # return word_vector

    return distance_vector


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


def choice_f(data, n_topic, n_words):
    lda_model, dictionary, corpus = lda.lda(data, n_topic)
    lda.print_coherence(lda_model, dictionary, corpus, data)
    lda.print_topics(lda_model, n_words)
    return


def choice_g(data, n_topic, n_words, year):
    start, stop, step = round((n_topic / 4) - 2), n_topic, 1
    lsa.plot_graph(data, start, stop, step, year)
    # model = lsa.create_gensim_lsa_model(data,n_topic,n_words)
    return


def printToFile(topicResults):
    with open('output/results.csv', 'w', encoding='UTF8') as csvfile:
        fieldnames = ['File', 'TopicWords', 'WordScore', 'TopicNumber']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, len(topicResults)):
            writer.writerow(
                {'File': topicResults[i][0], 'TopicWords': topicResults[i][1], 'WordScore': topicResults[i][2],
                 'TopicNumber': topicResults[i][3]})


if __name__ == "__main__":

    # print(sys.argv[0])  # prints python_script.py
    # print(sys.argv[1])  # prints var1 choose option
    # print(sys.argv[2])  # prints var2 year
    # print(sys.argv[3])  # prints var3 num cores

    if (len(sys.argv) == 4):
        arg_from_command_line = True
    else:
        arg_from_command_line = False

    while 1:

        if (arg_from_command_line == False):
            choose = input('Insert:\n'
                           'a) if you want the frequency of each nouns of each files\n'
                           'b) If you want the centroid of the densest area of the cluster\n'
                           'c) If you want to see the most frequent words of the cluster\n'
                           'd) If you want to see the most frequent words of the densest part of the cluster\n'
                           'e) If you want use top2vec to detect macro topics on all documents\n'
                           'f) lda\n'
                           'g) lsa\n')
        else:
            choose = str(sys.argv[1])

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
        elif choose == "bc":
            break
        elif choose == "f":
            break
        elif choose == "g":
            break

    # Working Folder
    os.chdir("data")
    listDoc = os.listdir()
    os.chdir("../")

    if choose != "e":
        # preprocess data
        if (arg_from_command_line == False):
            year = input("Insert year to be analyze: \n(insert skip if you want to scan all the documents)\n")
        else:
            year = str(sys.argv[2])
        # take all the names of the files
        filtered_docs_list = []
        all_docs = []

        for doc in listDoc:
            if doc.endswith(".txt"):
                all_docs.append(doc)
                # input_file = open(f"data/{doc}", encoding="utf8")
                # file_text = input_file.read()
                # all_docs.append(file_text)
            if doc.endswith(".txt") and year in doc:
                filtered_docs_list.append(doc)

        if filtered_docs_list == [] and year != "skip":
            print("No documents found for this decade")
            exit()

        if year == "skip":
            filtered_docs_list = all_docs

        if (arg_from_command_line == False):
            print("You have ", multiprocessing.cpu_count(), " cores")
            core_number = input('How many core do you want to use?: (Do not overdo it)\n')
        else:
            core_number = str(sys.argv[3])

        logger.info("Start Time : %s", datetime.now())
        start_time = datetime.utcnow()

        pool = multiprocessing.Pool(processes=int(core_number))  # creation of the pool of processes

        results = [pool.map(parallelized_function, filtered_docs_list)]  # array of documents preprocessed

        pool.close()  # close the pool of processes

        frequency_list = []

        if choose == "a" or choose == "bc":
            header = ['file_name', 'word_frequency']
            if year == "skip":
                title_header = ""
            else:
                title_header = year + "_"
            with open(f'output/{title_header}file_word_frequency.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)

            for i in range(len(filtered_docs_list)):
                with open(f'output/{title_header}file_word_frequency.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    # write file words
                    frequency_list.append(tp.word_count(results[0][i]))
                    data = [filtered_docs_list[i],
                            str(frequency_list[i]).replace(",", "").replace("[", "").replace("]", "")]
                    writer.writerow(data)

        # logs the time of the process
        logger.info("End Time : %s", datetime.now())
        end_time = datetime.utcnow()
        total_time = end_time - start_time
        logger.info("Total Time : %s", total_time)

        concat_results = np.concatenate(results[0])  # concat all the processed documents
        # concat_results_copy = concat_results

        # tag cloud of most frequent words of the decade
        if choose == "c" or choose == "bc":
            frequency = tp.word_count(concat_results)
            with open(f'output/{year}_WordFrequency.csv', 'w', encoding='UTF8') as f:
                mywriter = csv.writer(f, delimiter='\n')
                mywriter.writerows([frequency])
            # tp.tag_cloud(concat_results)
        if choose == "d":
            clear_results = [list(dict.fromkeys(concat_results))]  # remove duplicates
            tot_vectors = {}
            for word in clear_results[0]:
                tot_vectors[str(word)] = ew.get_embedding(str(word))
            choice_d(tot_vectors,
                     clear_results[0])  # tag cloud of most frequent words of the densest part of the decade
        if choose == "b" or choose == "bc":
            clear_results = [list(dict.fromkeys(concat_results))]  # remove duplicates
            tot_vectors = {}
            for word in clear_results[0]:
                tot_vectors[str(word)] = ew.get_embedding(str(word))
            topWords = choice_b(tot_vectors, year)[:50]  # get the centroid of the densest area of the cluster
            # print(topWords)

            # print results of the centroid of the densest area of the cluster in file
            with open(f'output/{year}_50TopWords.csv', 'w', encoding='UTF8') as f:
                mywriter = csv.writer(f, delimiter='\n')
                mywriter.writerows([topWords])
        if choose == "bc":
            file_score = []
            for i in range(len(filtered_docs_list)):
                counter = 0
                sum = 0
                for word, value in frequency_list[i]:
                    if word in topWords:
                        sum += value
                        counter += 1
                if counter != 0:
                    file_score.append([filtered_docs_list[i], counter * 100 / 50, sum / counter])
                else:
                    file_score.append([filtered_docs_list[i], 0, 0])
            file_score.sort(key=lambda x: x[1], reverse=True)
            with open(f'output/{year}_scores.csv', 'w', encoding='UTF8', newline='') as f:
                mywriter = csv.writer(f)
                mywriter.writerows(file_score)
        if choose == "f":
            choice_f(results[0], round(len(filtered_docs_list) / 2), 10)
        if choose == "g":
            choice_g(results[0],len(filtered_docs_list),10,year)
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
        for i in range(0, len(year_list) - 4, 5):
            year_list_10.append(year_list[i:i + 5])
        year_list_10.append(year_list[len(year_list) - len(year_list) % 5:])  # take the remaining years

        resultsForFile = []

        for group in year_list_10:
            list_files = []
            for year in group:
                for doc in listDoc:
                    if doc.endswith(".txt") and year in doc:
                        input_file = open(f"data/{doc}", encoding="utf8")
                        file_text = input_file.read()
                        list_files.append(file_text)
            logger.info("Start Top2Vec analysis for documents contained in the year: %s. Number of documents: %s",
                        group, len(list_files))
            topic_words, word_scores, topic_nums = choice_e(list_files)
            partial_results = [group, topic_words, word_scores, topic_nums]
            logger.info("End Top2Vec analysis for documents contained in the year: %s", group)
            resultsForFile.append(partial_results)

        printToFile(resultsForFile)
