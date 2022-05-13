import csv
import operator
import text_preprocessing as tp
if __name__ == "__main__":
    file_text =[]
    with open('5TopWords.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            for i in range (0, len(row)):
                file_text.append(row[i])
    frequence = {}
    for word in file_text:
        frequence[word]= 0

    for word in file_text:
        frequence[word] = frequence[word] +1
    frequence = sorted(frequence.items(), key=operator.itemgetter(1),
                                  reverse=True)
    print(frequence)
    tp.tag_cloud(file_text)
