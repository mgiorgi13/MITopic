import csv
import mit_topics
import embedding_word as ew

if __name__ == "__main__":
    # year = input("Insert year to be analyze: \n")

    for year in range(1990, 1993):
        fileToScan = []
        TopWords = []

        # read first 3 line from csv file
        with open(f'output/{year}_scores.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            counter = 0
            for row in csv_reader:
                if counter < 3:
                    fileToScan.append(row[0])
                counter += 1

        # extract 50 top words from each 3 file
        processed_file = []
        for file in fileToScan:
            processed_file.append(mit_topics.parallelized_function(file))

        for procFile in processed_file:
            tot_vectors = {}
            for word in procFile:
                tot_vectors[str(word)] = ew.get_embedding(str(word))
            topWords = mit_topics.choice_b(tot_vectors, year)[:50]
            with open(f'test/{year}_50TopWordsSINGFILE.csv', 'a+', encoding='UTF8') as f:
                mywriter = csv.writer(f, delimiter='\n')
                mywriter.writerows([topWords])
                mywriter.writerow(['\n-------------------------------------------\n'])
