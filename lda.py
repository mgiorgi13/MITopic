import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from pprint import pprint

def lda(data, num_topics):
    """
    LDA model.
    :param num_topics: Number of topics.
    :return: LDA model.
    """
    # Create Dictionary
    dictionary = corpora.Dictionary(data)
    # Create Corpus
    texts = data
    # Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in texts]

    #train LDA model
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        workers=8)
    return lda_model, dictionary, corpus


#print coherence of LDA model
def print_coherence(lda_model, dictionary, corpus, texts):
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

#print number of topics of LDA model
def print_topics(lda_model, num_words):
    print('\nTopics in LDA model:')
    # pprint(lda_model.print_topics())
    for idx, topic in lda_model.print_topics(num_words=num_words):
        print('Topic: {} \nWords: {}'.format(idx, topic))


