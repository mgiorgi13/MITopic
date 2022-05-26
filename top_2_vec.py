from top2vec import Top2Vec

# if This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU
# instructions in performance-critical operations:  AVX2 FMA To enable them in other operations, rebuild TensorFlow
# with the appropriate compiler flags. then
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def top_2_vec(documents):
    # training model with all documents.  speed : fast-learn / learn / deep-learn
    model = Top2Vec(documents, embedding_model='universal-sentence-encoder', speed='deep-learn', workers=4)
    numberOfTopis = model.get_num_topics()
    print(f"Number of topics : {numberOfTopis}")
    topic_words, word_scores, topic_nums = model.get_topics(numberOfTopis)
    print(f"Topic words : {topic_words}\nWord scores : {word_scores}\nTopic numbers : {topic_nums}")
    for topic in topic_nums:
        model.generate_topic_wordcloud(topic)
    return topic_words, word_scores, topic_nums