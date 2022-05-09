from typing import Text

import spacy

nlp = spacy.load('en_core_web_lg')


# python -m spacy download en_core_web_lg

def get_embedding(text: Text):
    nlp_text = nlp(text)

    embedding = nlp_text.vector
    norm_embedding = nlp_text.vector_norm
    return embedding
