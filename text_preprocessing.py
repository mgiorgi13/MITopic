# Import Module
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
           "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone",
           "twentytwo", "twentythree", "twentyfour", "twentyfive", "twentysix", "twentyseven", "twentyeight",
           "twentynine", "thirty","thirtyone", "thirtytwo", "thirtythree", "thirtyfour", "thirtyfive", "thirtysix",
           "thirtyseven", "thirtyeight", "thirtynine", "forty","fortyone", "fortytwo", "fortythree", "fortyfour",
           "fortyfive", "fortysix", "fortyseven", "fortyeight", "fortynine", "fifty","fiftyone", "fiftytwo",
           "fiftythree", "fiftyfour", "fiftyfive", "fiftysix", "fiftyseven", "fiftyeight", "fiftynine",
           "sixty","sixtyone", "sixtytwo", "sixtythree", "sixtyfour", "sixtyfive", "sixtysix", "sixtyseven",
           "sixtyeight", "sixtynine", "seventy","seventyone", "seventytwo", "seventythree", "seventyfour", "seventyfive",
           "seventysix", "seventyseven", "seventyeight", "seventynine", "eighty","eightyone", "eightytwo", "eightythree",
           "eightyfour", "eightyfive", "eightysix", "eightyseven", "eightyeight", "eightynine", "ninety","ninetyone",
           "ninetytwo", "ninetythree", "ninetyfour", "ninetyfive", "ninetysix", "ninetyseven", "ninetyeight",
           "ninetynine", "onehundred", "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five",
           "twenty-six", "twenty-seven", "twenty-eight", "twenty-nine", "thirty-one", "thirty-two", "thirty-three",
           "thirty-four", "thirty-five", "thirty-six", "thirty-seven", "thirty-eight", "thirty-nine", "forty-one",
           "forty-two", "forty-three", "forty-four", "forty-five", "forty-six", "forty-seven", "forty-eight",
           "forty-nine", "fifty-one", "fifty-two", "fifty-three", "fifty-four", "fifty-five", "fifty-six",
           "fifty-seven", "fifty-eight", "fifty-nine", "sixty-one", "sixty-two", "sixty-three", "sixty-four",
           "sixty-five", "sixty-six", "sixty-seven", "sixty-eight", "sixty-nine", "seventy-one", "seventy-two",
           "seventy-three", "seventy-four", "seventy-five", "seventy-six", "seventy-seven", "seventy-eight",
           "seventy-nine", "eighty-one", "eighty-two", "eighty-three", "eighty-four", "eighty-five", "eighty-six",
           "eighty-seven", "eighty-eight", "eighty-nine", "ninety-one", "ninety-two", "ninety-three", "ninety-four",
           "ninety-five", "ninety-six", "ninety-seven", "ninety-eight", "ninety-nine", "one-hundred"]
punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
               '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

lemmatizer = WordNetLemmatizer()

stopwords_en = stopwords.words('english')
stopwords_en.extend(numbers)
stopwords_en.extend(punctuation)


def print_stopword():
    print(stopwords_en)
    print(len(stopwords_en))


def remove_whitespace(text):
    return " ".join(text.split())


def tokenization(text):
    # lower-casing
    lower_text = text.lower()
    # list of token
    return nltk.word_tokenize(lower_text)


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


def checkInt(str):
    try:
        int(str)
        return True
    except ValueError:
        return False


def pos_tagging(stopword_cleaned):
    pos_tagging = nltk.pos_tag(stopword_cleaned)

    cleaned_POS_text = []

    for tuple in pos_tagging:
        # POS tagged text is a list of tuples, where the first element tuple[0] is a token and the second one tuple[1] is
        # the Part of Speech. If the POS has length == 1, the token is punctuation, otherwise it is not, and we insert it
        # in the list cleaned_POS_text
        if len(tuple[1]) > 1 and tuple[0] != '’' and tuple[0] != '“' and tuple[0] != '”' and checkInt(
                tuple[0][0]) == False:
            cleaned_POS_text.append(tuple)

    simpler_POS_text = []

    # for each tuple of the list, we create a new tuple: the first element is the token, the second is
    # the simplified pos tag, obtained calling the function simpler_pos_tag()
    # then we append the new created tuple to a new list, which will be the output
    for tuple in cleaned_POS_text:
        POS_tuple = (tuple[0], simpler_pos_tag(tuple[1]))
        if (POS_tuple[1] == "n"):
            simpler_POS_text.append(POS_tuple)

    return simpler_POS_text


def lemmatization(simpler_POS_text):
    lemmatized_text = []

    for tuple in simpler_POS_text:
        if (tuple[1] == None):
            lemmatized_text.append(lemmatizer.lemmatize(tuple[0]))
        else:
            lemmatized_text.append(lemmatizer.lemmatize(tuple[0], pos=tuple[1]))

    return lemmatized_text
