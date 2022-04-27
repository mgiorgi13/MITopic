# Import Module
import os
import text_preprocessing as tp

from tmtoolkit.corpus import Corpus, tokens_table, lemmatize, to_lowercase, dtm
from tmtoolkit.bow.bow_stats import tfidf, sorted_terms_table

if __name__ == "__main__":
    # Working Folder
    os.chdir("data")
    count = 0
    for file in os.listdir():
        count = count + 1

        if file.endswith(".txt"):
            input_file = open(file, encoding="utf8")
            file_text = input_file.read()

            file_text = tp.remove_whitespace(file_text)
            file_text = tp.tokenization(file_text)
            file_text = tp.stopword_removing(file_text)
            file_text = tp.pos_tagging(file_text)
            file_text = tp.lemmatization(file_text)

            print(tp.word_count(file_text))
            tp.tag_cloud(file_text)

        if(count == 10):
            break

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