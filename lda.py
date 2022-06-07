import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import os

def lda(data, num_topics, year):
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
    if not os.path.exists(f"output/{year}/LDA"):
        os.makedirs(f"output/{year}/LDA")
    print_lda_results(lda_model, dictionary, corpus, texts, 'lda_results.txt', year)
    # print_documents_results(lda_model, corpus, 'documents_results.txt', year) # not working
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


# print documents results into a file.txt
def print_documents_results(lda_model, corpus, filename, year):
    with open(f"output/{year}/LDA/{filename}", 'w') as f:
        f.write('Documents in LDA model:\n')
        for i, topic in lda_model[corpus]:
            f.write('Document: {} \nTopic: {}\n'.format(i, topic))


# print lda results into a file.txt
def print_lda_results(lda_model, dictionary, corpus, texts, filename, year):
    with open(f"output/{year}/LDA/{filename}", 'w') as f:
        f.write('Topics in LDA model:\n')
        for idx, topic in lda_model.print_topics(num_words=10):
            f.write('Topic: {} \nWords: {}\n'.format(idx, topic))
        f.write('\nCoherence Score: {}'.format(
            CoherenceModel(
                model=lda_model,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v').get_coherence()))