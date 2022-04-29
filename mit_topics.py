# Import Module
import os
import text_preprocessing as tp
import top_2_vec as tv

from tmtoolkit.corpus import Corpus, tokens_table, lemmatize, to_lowercase, dtm
from tmtoolkit.bow.bow_stats import tfidf, sorted_terms_table

if __name__ == "__main__":
    # Working Folder
    os.chdir("data")
    count = 0
    documents = []

    for file in os.listdir():
        count = count + 1

        if file.endswith(".txt"):
            input_file = open(file, encoding="utf8")
            file_text = input_file.read()

            # file_text = tp.remove_whitespace(file_text)
            # file_text = tp.tokenization(file_text)
            # file_text = tp.stopword_removing(file_text)
            # file_text = tp.pos_tagging(file_text)
            # file_text = tp.lemmatization(file_text)

            documents.append(file_text)

            # top_word = tp.word_count(file_text)
            # document = ""
            # for words in top_word:
            #     if(words[1] < 5):
            #         break
            #     document = document + words[0] + " "
            # documents.append(document)

            # print(tp.word_count(file_text))
            # tp.tag_cloud(file_text)

    tv.top_2_vec(documents)



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