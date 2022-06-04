import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel


def lda(data, num_topics):
    """
    LDA model.
    :param data: documents to be analyzed.
    :param num_topics: Number of topics.
    :return: LDA model.
    """
    # Create Dictionary
    dictionary = corpora.Dictionary(data)
    # Create Corpus
    texts = data
    # Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in texts]

    # train LDA model
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        alpha='auto',
        eta='auto',
        passes=10)
    return lda_model, dictionary, corpus


# print coherence of LDA model
def print_coherence(lda_model, dictionary, corpus, texts):
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


# print topics of LDA model
def print_topics(lda_model, num_words):
    print('\nTopics in LDA model:')
    for idx, topic in lda_model.print_topics(num_words=num_words):
        print('Topic: {} \nWords: {}'.format(idx, topic))


# # print documents of LDA model
# # TODO actually this function is not working
# def print_documents(lda_model, corpus):
#     print('\nDocuments in LDA model:')
#     for i, topic in lda_model[corpus]:
#         print('Document: {} \nTopic: {}'.format(i, topic))
