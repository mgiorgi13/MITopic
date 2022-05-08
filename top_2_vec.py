from top2vec import Top2Vec

def top_2_vec(documents):
    #speed : fast-learn / learn / deep-learn
    model = Top2Vec(documents, embedding_model='universal-sentence-encoder', speed='deep-learn', workers=4)
    numberOfTopis = model.get_num_topics()
    print(f"Number of topics : {numberOfTopis}")
    topic_words, word_scores, topic_nums = model.get_topics(numberOfTopis)
    print(f"Topic words : {topic_words}\nWord scores : {word_scores}\nTopic numbers : {topic_nums}")
    for topic in topic_nums:
        model.generate_topic_wordcloud(topic)