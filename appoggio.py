# Import Module
import os
import text_preprocessing as tp
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import gensim
import top_2_vec as tv


# from tmtoolkit.corpus import Corpus, tokens_table, lemmatize, to_lowercase, dtm
# from tmtoolkit.bow.bow_stats import tfidf, sorted_terms_table
def TFIDFScore(simple_text):
    cv = CountVectorizer()
    # this steps generates word counts for the words in your docs
    word_count_vector = cv.fit_transform(simple_text)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    # print idf values
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
    # sort ascending
    df_idf.sort_values(by=['idf_weights'])
    # count matrix
    count_vector = cv.transform(simple_text)
    # tf-idf scores
    tf_idf_vector = tfidf_transformer.transform(count_vector)
    feature_names = cv.get_feature_names()
    # get tfidf vector for first document
    first_document_vector = tf_idf_vector[0]
    # print the scores
    df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)


def Tfidfvectorizer(simple_text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # settings that you use for count vectorizer will go here
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    # just send in all your docs here
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(simple_text)
    # get the first vector out (for the first document)
    first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]
    # place tf-idf values in a pandas data frame
    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
                      columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)
    from sklearn.cluster import KMeans
    modelkmeans = KMeans(n_clusters=60, init='k-means++', n_init=100)
    modelkmeans.fit(df)
    from sklearn.cluster import KMeans

    Sum_of_squared_distances = []
    K = range(1, 100)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(df)
        Sum_of_squared_distances.append(km.inertia_)

    import matplotlib.pyplot as plt

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    return


if __name__ == "__main__":
    # Working Folder
    os.chdir("data")
    count = 0
    documents = []
    sentences = []
    for file in os.listdir():
        count = count + 1

        if file.endswith(".txt"):
            input_file = open(file, encoding="utf8")
            file_text = input_file.read()

            file_text = tp.remove_whitespace(file_text)  # rimozione doppi spazi
            file_text = tp.tokenization(file_text)  # tokenizzo
            file_text = tp.stopword_removing(file_text)  # rimuovo le stopword
            file_text = tp.pos_tagging(file_text)  # metto un tag ad ogni parola
            file_text = tp.lemmatization(file_text)  # trasformo nella forma base ogni parola
            tp.tag_cloud(file_text)  # stamo in base alla frequeza di ogni parola
            # TFIDFScore(file_text)
            # Tfidfvectorizer(file_text)
            break

    #        sentences.extend(file_text)

    # #instantiete the model and train it
    # word2vec_model = gensim.models.Word2vec(
    #     sentences,
    #     sg = 1, # here 1 will use  skipgram (0 in  CBOW). skipgrams for small corpora
    #     min_count = 1 #word need to accur  at least once
    # )
    #
    # #update the model
    #
    # print("ciao")
    # documents.append(file_text)

    # top_word = tp.word_count(file_text)
    # document = ""
    # for words in top_word:
    #     if(words[1] < 5):
    #         break
    #     document = document + words[0] + " "
    # documents.append(document)

    # print(tp.word_count(file_text))
    # tp.tag_cloud(file_text)

    # tv.top_2_vec(documents)

    # # load built-in sample dataset and use 4 worker processes
    # corp = Corpus.from_builtin_corpus('en-News100', max_workers=4)
    # # investigate corpus as dataframe
    # toktbl = tokens_table(corp)
    # print(toktbl)
    # # apply some text normalization
    # lemmatize(corp)
    # to_lowercase(corp)
    # # build sparse document-token matrix (DTM)
    # # document labels identify rows, vocabulary tokens identify columns
    # mat, doc_labels, vocab = dtm(corp, return_doc_labels=True, return_vocab=True)
    # # apply tf-idf transformation to DTM
    # # operation is applied on sparse matrix and uses few memory
    # tfidf_mat = tfidf(mat)
    # # show top 5 tokens per document ranked by tf-idf
    # top_tokens = sorted_terms_table(tfidf_mat, vocab, doc_labels, top_n=5)
    # print(top_tokens)
