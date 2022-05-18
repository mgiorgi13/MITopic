
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
    #  pca.pca_clustering_3D(transformer.transform(value_vactor), list(tot_vectors.keys()), f"/html/{file[: -4]}")

    sortedDist = ct.centroid_Topic(transformer.transform(value_vactor), word_vector)
    # print(sortedDist)
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

def choice_e(list_files):
    documents = []
    for file in list_files:
        if file.endswith(".txt"):
            input_file = open(f"data/{file}", encoding="utf8")
            file_text = input_file.read()
            documents.append(file_text)
    t2v.top_2_vec(documents)








    if choose == "b":

        print("You have ", multiprocessing.cpu_count(), " cores")
        core_number = input('How many core do you want to use?: (Do not overdo it)\n')

        logger.info("Start Time : %s", datetime.now())
        start_time = datetime.utcnow()

        pool = multiprocessing.Pool(processes=int(core_number))

        # limit listDoc for test with listDoc[0:m] with m = number of documents you want to test
        # delete [0:m] if you want to test all documents
        if decade == "skip":
            results = [pool.map(parallelized_function, listDoc)]


            logger.info("End Time : %s", datetime.now())
            pool.close()

            with open('output/5TopWords.csv', 'w') as f:
                mywriter = csv.writer(f, delimiter='\n')
                mywriter.writerows(results)

            end_time = datetime.utcnow()
            total_time = end_time - start_time
            logger.info("Total Time : %s", total_time)
        else:
            results = [pool.map(parallelized_function, filtered_docs_list)]

            concat_results = np.concatenate(results[0])
            concat_results = [list(dict.fromkeys(concat_results))]

            tot_vectors = {}
            for word in concat_results:
                tot_vectors[str(word)] = ew.get_embedding(str(word))

            topWords = choice_b(tot_vectors)[:30]

            logger.info("End Time : %s", datetime.now())
            pool.close()

            with open(f'output/{decade}_5TopWords.csv', 'w') as f:
                mywriter = csv.writer(f, delimiter='\n')
                mywriter.writerows(concat_results)

            end_time = datetime.utcnow()
            total_time = end_time - start_time
            logger.info("Total Time : %s", total_time)

    elif choose == "e":
       choice_e(listDoc)
    else:
        for file in tqdm(listDoc):
            count = count + 1

            if file.endswith(".txt"):
                input_file = open(f"data/{file}", encoding="utf8")
                file_text = input_file.read()

                if choose != "e":
                    file_text = tp.remove_whitespace(file_text)  # rimozione doppi spazi
                    file_text = tp.tokenization(file_text)  # tokenizzo
                    file_text = tp.stopword_removing(file_text)  # rimuovo le stopword
                    file_text = tp.pos_tagging(file_text)  # metto un tag ad ogni parola
                    file_text = tp.lemmatization(file_text)  # trasformo nella forma base ogni parola

                    tot_vectors = {}

                    for word in (file_text):
                        tot_vectors[word] = ew.get_embedding(word)

                if choose == "a":
                    choice_a(tot_vectors)
                    break
                # elif choose == "b":
                #     if count == 1:
                #         cc.write_list_as_row("5TopWords.csv", choice_b(tot_vectors)[:5])
                #     else:
                #         cc.append_list_as_row("5TopWords.csv", choice_b(tot_vectors)[:5])
                # # break
                elif choose == "c":
                    choice_c(file_text)
                    break
                elif choose == "d":
                    choice_d(tot_vectors, file_text)
                    break
