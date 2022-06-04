import csv
import logging
import coloredlogs
import os
from datetime import datetime
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
import text_preprocessing as tp
import embedding_word as ew
import PCA_plot3D as pca
import operator



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
        transformer = RobustScaler(quantile_range=( 0, 75.0))
        transformer.fit(list(tot_vectors.values()))
        centroid_ = transformer.center_
        centroid_ = np.array([centroid_])
        distance_vector = {}
        for j in range(0, len(tot_vectors) - 1):
            dist = cosine_similarity(centroid_, np.array([list(tot_vectors.values())[j]]))
            distance_vector[list(tot_vectors.keys())[j]] = dist[0][0]
        distance_vector = sorted(distance_vector.items(), key=operator.itemgetter(1),
                                      reverse=True)


        dct = {}
        dct[1] = []
        dct[-1] = []
        dct[0] = []
        for s in range(0, len(distance_vector)):
            if distance_vector[s][1]<= 1 and distance_vector[s][1]  > 0.3:
                dct[1].append(distance_vector[s][0])
                continue
            if distance_vector[s][1] <= 0.3 and distance_vector[s][1] > -0.5:
                dct[0].append(distance_vector[s][0])
                continue
            if distance_vector[s][1] <= -0.5 and distance_vector[s][1] >= -1:
                dct[-1].append(distance_vector[s][0])
                continue
        path = f"output/{year}/{title[i][:-4]}"
        if os.path.exists(path) == False:
            os.makedirs(path)
        with open(f"{path}/{year}_TopWords.txt", "w") as f:
            f.write("1:")
            f.write(" \n")
            for word in dct[1]:
                f.write(word + ", ")
            f.write(" \n")
            f.write(" \n")
            f.write("0:")
            f.write(" \n")
            for word in dct[0]:
                f.write(word + ", ")
            f.write(" \n")
            f.write(" \n")
            f.write("-1:")
            f.write(" \n")
            for word in dct[-1]:
                f.write(word + ", ")
            f.write(" \n")

            words = []
            for p in range(0, len(docs[i])):
                for t in range(0, len(dct[1])):
                    if dct[1][t] == docs[i][p]:
                        words.append(dct[1][t])

            tp.tag_cloud(words, year)

def STD():

    # print(sys.argv[0])  # prints python_script.py
    # print(sys.argv[1])  # prints var1 choose option
    # print(sys.argv[2])  # prints var2 year
    # print(sys.argv[3])  # prints var3 num cores

    year = input("Insert year to be analyze: \n(insert skip if you want to scan all the documents)\n")


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