# Import Module
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stopwords_en = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def print_stopword():
    print(stopwords_en)

def remove_whitespace(text):
    return " ".join(text.split())

def tokenization(text):
    # lower-casing
    lower_text = text.lower()
    # the output is a list, where each element is a sentence of the original text
    nltk.sent_tokenize(lower_text)
    # the output is a list, where each element is a token of the original text
    tokenized_text = nltk.word_tokenize(lower_text)
    return tokenized_text

def stopword_removing(tokenized_text):
    # we prepare an empty list, which will contain the words after the stopwords removal
    stopword_cleaned = []

    # we iterate into the list of tokens obtained through the tokenization
    for token in tokenized_text:
        # if a token is not a stopword, we insert it in the list
        if token not in stopwords_en:
            stopword_cleaned.append(token)

    # the output is a list of all the tokens of the original text excluding the stopwords
    return stopword_cleaned

def simpler_pos_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return "a"
    elif nltk_tag.startswith('V'):
        return "v"
    elif nltk_tag.startswith('N'):
        return "n"
    elif nltk_tag.startswith('R'):
        return "r"
    else:
        return None

def pos_tagging(stopword_cleaned):

    pos_tagging = nltk.pos_tag(stopword_cleaned)

    cleaned_POS_text = []

    for tuple in pos_tagging:
        # POS tagged text is a list of tuples, where the first element tuple[0] is a token and the second one tuple[1] is
        # the Part of Speech. If the POS has length == 1, the token is punctuation, otherwise it is not, and we insert it
        # in the list cleaned_POS_text
        if len(tuple[1]) > 1:
            cleaned_POS_text.append(tuple)

    print(cleaned_POS_text)

    simpler_POS_text = []

    # for each tuple of the list, we create a new tuple: the first element is the token, the second is
    # the simplified pos tag, obtained calling the function simpler_pos_tag()
    # then we append the new created tuple to a new list, which will be the output
    for tuple in cleaned_POS_text:
        POS_tuple = (tuple[0], simpler_pos_tag(tuple[1]))
        simpler_POS_text.append(POS_tuple)

    print(simpler_POS_text)

    return simpler_POS_text

def lemmatization(simpler_POS_text):
    lemmatized_text = []

    for tuple in simpler_POS_text:
        if (tuple[1] == None):
            lemmatized_text.append(lemmatizer.lemmatize(tuple[0]))
        else:
            lemmatized_text.append(lemmatizer.lemmatize(tuple[0], pos=tuple[1]))

    print(lemmatized_text)
    return lemmatized_text