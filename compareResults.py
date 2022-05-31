import csv
import json


def lostWords(currentWords, previousWords):
    lostWords = []
    for prevWord in previousWords:
        if prevWord not in currentWords:
            lostWords.append(prevWord)
    return lostWords


def gainWords(currentWords, previousWords):
    gainWords = []
    for currWord in currentWords:
        if currWord not in previousWords:
            gainWords.append(currWord)
    return gainWords


def equalWords(currentWords, previousWords):
    equalWords = []
    for prevWord in previousWords:
        if prevWord in currentWords:
            equalWords.append(prevWord)
    return equalWords


def compareTop50Words(year):
    if year == 1990:
        currentTop50 = []
        with open(f'output/{year}_50TopWords.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) != 0:
                    currentTop50.append(row[0])
        return [], [], [], currentTop50
    else:
        currentTop50 = []
        previousTop50 = []
        with open(f'output/{year}_50TopWords.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) != 0:
                    currentTop50.append(row[0])
        with open(f'output/{year - 1}_50TopWords.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) != 0:
                    previousTop50.append(row[0])

        gain = gainWords(currentTop50, previousTop50)
        lost = lostWords(currentTop50, previousTop50)
        equal = equalWords(currentTop50, previousTop50)
    return gain, lost, equal, currentTop50


# def mergeWordFrq():
#     for year in range(1990, 2023):
#         top50 = []
#         freqWord = []
#         print(year)
#         # read 50 top words
#         with open(f'output/{year}_50TopWords.csv') as csv_file:
#             csv_reader = csv.reader(csv_file, delimiter=',')
#             for row in csv_reader:
#                 if len(row) != 0:
#                     top50.append(row[0])
#
#         # read frequency of words
#         with open(f'output/{year}_WordFrequency.csv') as csv_file:
#             csv_reader = csv.reader(csv_file, delimiter=',')
#             for row in csv_reader:
#                 if len(row) != 0:
#                     freqWord.append(row)
#         # parse frequency of words
#         for elem in freqWord:
#             if len(elem) != 1:
#                 word = elem[0].replace("(", "").replace("('", "").replace("'", "").replace("\o", "").replace('"', "")
#                 elem[0] = word
#                 freq = elem[1].replace(")", "")
#                 elem[1] = int(freq)
#             else:
#                 print("sono qui ")
#                 word = elem[0].replace("(", "").replace("('", "").replace("'", "").replace("\o", "").replace('"', "")
#                 word = word.split(",")
#                 elem[0] = word[0]
#                 freq = word[1].replace(")", "")
#                 elem.append(int(freq))
#
#         top50freq = []
#
#         # find frequency of top 50 words
#         for elem in top50:
#             for word in freqWord:
#                 if elem == word[0]:
#                     top50freq.append([word[0], word[1]])
#
#         top50freq.sort(key=lambda x: x[1], reverse=True)
#
#         # write top 50 words and frequency to file
#         with open(f'output/{year}_Top50WordsFrequency.csv', 'w') as csv_file:
#             csv_writer = csv.writer(csv_file, delimiter=',')
#             for elem in top50freq:
#                 csv_writer.writerow(elem)


def findFreq(year, word):
    with open(f'output/{year}_Top50WordsFrequency.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) != 0:
                if row[0] == word:
                    return row[1]


if __name__ == "__main__":
    # mergeWordFrq() # merge top 50 words and frequency

    data = {}  # json data structure with all years
    list = []  # list of all years
    for year in range(1990, 2023):
        gain, lost, equal, currentTop50 = compareTop50Words(year)
        jsonCurrent = []
        jsonGain = []
        jsonLost = []
        jsonEqual = []
        for elem in currentTop50:
            jsonCurrent.append({"word": elem, "frequency": findFreq(year, elem)})
        for elem in gain:
            jsonGain.append({"word": elem, "frequency": findFreq(year, elem)})
        for elem in lost:
            jsonLost.append({"word": elem, "frequency": findFreq(year - 1, elem)})
        for elem in equal:
            jsonEqual.append({"word": elem, "frequency": findFreq(year, elem)})
        elem = {"year": year, "currentTop50": jsonCurrent, "gain": jsonGain, "lost": jsonLost, "equal": jsonEqual}
        list.append(elem)

    data["years"] = list  # append list to json data structure
    # print json to file
    with open('output/compareResults.json', 'w') as outfile:
        json.dump(data, outfile)
