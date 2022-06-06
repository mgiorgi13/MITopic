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
import result_visualization as rv

# GLOBAL VARIABLES
from STD import STD

choose = ""

'''
    multiprocess function to preprocess the text
'''

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                    fmt="- Process -> pid[%(process)d], name[%(processName)s] Function -> [%(funcName)s]\n%(asctime)s --- %(levelname)s log -> [%(message)s]")

best_topic_number = {}

with open('best_num_topic.txt', 'r', encoding='UTF8') as file:
    for line in file:
        text = line
        text = text.split(" ")
        best_topic_number[text[0]] = text[1]


def parallelized_function(file):  # use to clear and prepare source text of each file
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


def choice_lda_method(data, n_topic, n_words, year):
    lda_model, dictionary, corpus = lda.lda(data, n_topic, year)
    # lda.print_coherence(lda_model, dictionary, corpus, data)
    # lda.print_topics(lda_model, n_words)
    # lda.print_documents(lda_model, n_words)
    return


def choice_lsa_method(data, n_topic, n_words, year):
    # # decomment to search for the best number of topics
    # start, stop, step = round((n_topic / 4) - 2), n_topic, 1
    # lsa.plot_graph(data, start, stop, step, year)
    model = lsa.create_gensim_lsa_model(data, n_topic, n_words, year)
    return


def frequency_analysis_file(year, filtered_docs_list, results):
    header = ['file_name', 'word_frequency']
    title_header = year + "_"
    if not os.path.exists(f"output/{year}/wordcloud"):
        os.makedirs(f"output/{year}/wordcloud")
    with open(f'output/{year}/wordcloud/{title_header}files_word_frequency.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # write the header
    frequency_list = []
    for i in range(len(filtered_docs_list)):
        with open(f'output/{year}/wordcloud/{title_header}files_word_frequency.csv', 'a', encoding='UTF8',
                  newline='') as f:
            writer = csv.writer(f)
            # write file words
            frequency_list.append(rv.word_count(results[0][i]))
            data = [filtered_docs_list[i],
                    str(frequency_list[i]).replace(",", "").replace("[", "").replace("]", "")]
            writer.writerow(data)
    return


def word_cloud_method(file_text, year):
    words = []
    for i in range(0, len(file_text)):
        words.append(file_text[i])
    rv.tag_cloud(words, year, "-")
    return


def choice_wordcloud_method(file_text, year, filtered_docs_list, results):
    # tag cloud of most frequent words of the year
    word_cloud_method(file_text, year)
    #histogram
    rv.histogram(year, file_text)
    # frequency analysis of the most frequent words of each file in the selected year
    frequency_analysis_file(year, filtered_docs_list, results)
    return


def choice_clustering_method(tot_vectors, year, file_text):
    bigClusters = db.DBSCAN_Topic(tot_vectors, year)
    words = []
    for i in range(0, len(file_text)):
        for t in range(0, len(bigClusters)):
            for j in range(0, len(bigClusters[t])):
                if bigClusters[t][j] == file_text[i]:
                    words.append(bigClusters[t][j])
    return


def choice_top2vec(list_files):
    topic_words, word_scores, topic_nums = t2v.top_2_vec(list_files)
    return topic_words, word_scores, topic_nums


def printToFile(topicResults, type, year):
    if type == "group":
        if not os.path.exists(f"output/top2vec/grouped"):
            os.makedirs(f"output/top2vec/grouped")
        with open(f'output/top2vec/grouped/grouped_results.csv', 'w', encoding='UTF8') as csvfile:
            fieldnames = ['File', 'TopicWords', 'WordScore', 'TopicNumber']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(0, len(topicResults)):
                writer.writerow(
                    {'File': topicResults[i][0], 'TopicWords': topicResults[i][1], 'WordScore': topicResults[i][2],
                     'TopicNumber': topicResults[i][3]})
    else:
        if not os.path.exists(f"output/top2vec/single/{year}"):
            os.makedirs(f"output/top2vec/single/{year}")
        with open(f'output/top2vec/single/{year}/{year}_results.csv', 'w', encoding='UTF8') as csvfile:
            fieldnames = ['File', 'TopicWords', 'WordScore', 'TopicNumber']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(0, len(topicResults)):
                writer.writerow(
                    {'File': topicResults[i][0], 'TopicWords': topicResults[i][1], 'WordScore': topicResults[i][2],
                     'TopicNumber': topicResults[i][3]})


if __name__ == "__main__":

    if (len(sys.argv) == 4):
        arg_from_command_line = True
    else:
        arg_from_command_line = False

    while 1:

        if (arg_from_command_line == False):
            choose = input('Insert:\n\n'
                           '- LDA -> Digit "lda" to execute this text mining method.\n\n'
                           '- LSA -> Digit "lsa" to execute this text mining method.\n\n'
                           '- TOP_2_VEC -> Digit "top2vec" to execute this text mining method.\n\n'
                           '- WORD_CLOUD -> Digit "wordcloud" to execute this text mining method.\n\n'
                           '- GTD -> Digit "gtd-cluster" to execute this text mining method in a year.\n\n'
                           '- STD -> Digit "std" to execute this text mining method for each file in a year. \n\n'
                           '- EXIT -> Digit "exit" to exit the program.\n\n')
        else:
            choose = str(sys.argv[1])

        if choose == 'lda' or choose == 'lsa' or choose == 'top2vec' or choose == 'wordcloud' or choose == 'gtd-cluster' or choose == 'pca' or choose == 'std':
            break
        else:
            exit()  # choose seected is exit

    # get folder with all the files to analyze
    # recommendation: put the files in the data folder and
    # make sure that each file has a title like string_string_year and saved in a .txt format
    os.chdir("data")
    listDoc = os.listdir()
    os.chdir("../")

    # extract the years from the file names
    years_list = []
    for doc in listDoc:
        year = doc.split("_")[2]
        if year not in years_list:
            years_list.append(year)
    years_list.sort()

    if choose == 'std':
        STD()
    else:
        if choose != "top2vec":  # if choose is not top2vec, becouse top2vec is the only method that can't analyze single year
            # get the year to analyze
            if (arg_from_command_line == False):
                year = input("Insert year to be analyze: \n""(you can choose from the list: " + str(years_list) + ")\n")
            else:
                year = str(sys.argv[2])

            # check if the year is in the list of years
            if year not in years_list:
                print("Year not in the list")
                exit()

            # take all the names of the files
            filtered_docs_list = []  # list of the files to analyze filtered by year
            all_docs = []  # list of all the files
            for doc in listDoc:
                if doc.endswith(".txt"):
                    all_docs.append(doc)
                if doc.endswith(".txt") and year in doc:
                    filtered_docs_list.append(doc)
            if filtered_docs_list == []:  # we found no files to analyze for the selected year
                print("No documents found for this decade")
                exit()

            # choose how many subprocess to use
            if arg_from_command_line == False:
                print("You have ", multiprocessing.cpu_count(), " cores")
                core_number = input('How many core do you want to use?: (Do not overdo it)\n')
            else:
                core_number = str(sys.argv[3])

            # create a pool of processes to clear the source text
            logger.info("Start Time : %s", datetime.now())
            start_time = datetime.utcnow()

            pool = multiprocessing.Pool(processes=int(core_number))  # creation of the pool of processes

            results = [pool.map(parallelized_function, filtered_docs_list)]  # array of documents preprocessed

            pool.close()  # close the pool of processes

            concat_results = np.concatenate(results[0])  # concat all the processed documents

            # logs the time of the total process about the selected year
            logger.info("End Time : %s", datetime.now())
            end_time = datetime.utcnow()
            total_time = end_time - start_time
            logger.info("Total Time : %s", total_time)

            if choose == "lda":
                choice_lda_method(results[0], int(best_topic_number[year]), 10, year)
            if choose == "lsa":
                choice_lsa_method(results[0], int(best_topic_number[year]), 10, year)
            if choose == "wordcloud":
                clear_results = [
                    list(dict.fromkeys(concat_results))]  # remove duplicates for clustering method
                tot_vectors = {}
                for word in clear_results[0]:
                    tot_vectors[str(word)] = ew.get_embedding(str(word))  # get the embedding of each word
                choice_wordcloud_method(concat_results, year, filtered_docs_list, results)
            if choose == "gtd-cluster":
                clear_results = [list(dict.fromkeys(concat_results))]  # remove duplicates for clustering method
                tot_vectors = {}
                for word in clear_results[0]:
                    tot_vectors[str(word)] = ew.get_embedding(str(word))  # get the embedding of each word
                # get the top 50 words of the most dense cluster
                if not os.path.exists(f"output/{year}/PCA/"):
                    os.makedirs(f"output/{year}/PCA/")
                pca.pca_clustering_3D(list(tot_vectors.values()), list(tot_vectors.keys()),
                                      f"output/{year}/PCA/InitialCluster__nWords_{len(tot_vectors)}")
                choice_clustering_method(tot_vectors, year, clear_results[0])
        else:  # choose is top2vec
            singleORgrouped = input("Do you want to analyze a single year or a group of years?\n"
                                    "-> Digit 'single' to analyze a single year (may not working for every years)\n"
                                    "-> Digit 'group' to analyze a group of years\n")
            if singleORgrouped == "single":
                print("single")
                year = input("Insert year to be analyze: \n""(you can choose from the list: " + str(years_list) + ")\n")
                # check if the year is in the list of years
                if year not in years_list:
                    print("Year not in the list")
                    exit()

                # take all the names of the files
                filtered_docs_list = []  # list of the files to analyze filtered by year
                for doc in listDoc:
                    if doc.endswith(".txt") and year in doc:
                        filtered_docs_list.append(doc)
                if filtered_docs_list == []:  # we found no files to analyze for the selected year
                    print("No documents found for this decade")
                    exit()

                list_files = []
                for doc in filtered_docs_list:
                    input_file = open(f"data/{doc}", encoding="utf8")
                    file_text = input_file.read()
                    list_files.append(file_text)
                logger.info("Start Top2Vec analysis for documents contained in the year: %s. Number of documents: %s",
                            year, len(list_files))
                topic_words, word_scores, topic_nums = choice_top2vec(list_files)
                partial_results = [year, topic_words, word_scores, topic_nums]
                logger.info("End Top2Vec analysis for documents contained in the year: %s", year)
                printToFile(partial_results, "single", year)
            else:
                # execute top_2_vec on documents grouped by ten years
                # extract interval of 10 year from the list of years
                year_list_10 = []
                for i in range(0, len(years_list) - 9, 10):
                    year_list_10.append(years_list[i:i + 10])
                year_list_10.append(years_list[len(years_list) - len(years_list) % 10:])  # take the remaining years

                resultsForFile = []

                for group in year_list_10:
                    list_files = []
                    for year in group:
                        for doc in listDoc:
                            if doc.endswith(".txt") and year in doc:
                                input_file = open(f"data/{doc}", encoding="utf8")
                                file_text = input_file.read()
                                list_files.append(file_text)
                    logger.info(
                        "Start Top2Vec analysis for documents contained in the year: %s. Number of documents: %s",
                        group, len(list_files))
                    topic_words, word_scores, topic_nums = choice_top2vec(list_files)
                    partial_results = [group, topic_words, word_scores, topic_nums]
                    logger.info("End Top2Vec analysis for documents contained in the year: %s", group)
                    resultsForFile.append(partial_results)

                printToFile(resultsForFile, "group", "")
