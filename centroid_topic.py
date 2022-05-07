import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import operator


def centroid_Topic(vector_words, word_vector):
    centroid = vector_words[0]
    for i in vector_words:
        centroid = (centroid + i)
    centroid = centroid / len(vector_words)

    centroid = np.array([centroid]);  # dava probelmi di dimensionalità

    distance_vector = {}
    # calcolo la distana che c'è tra centroide e valore del  di ogni signola parola
    for i in range(0, len(vector_words) - 1):
        dist = cosine_similarity(centroid, np.array([vector_words[i]]))
        distance_vector[word_vector[i]] = dist[0][0]

    return sorted(distance_vector.items(), key=operator.itemgetter(-1), reverse=True)
