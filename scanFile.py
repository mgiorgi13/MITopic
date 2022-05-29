import csv

if __name__ == "__main__":
    # year = input("Insert year to be analyze: \n")

    for year in range(1990, 2022):
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

        # read all lines from csv file
        with open(f'output/{year}_50TopWords.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                TopWords.append(row)

        # read 3 file.txt and extract significant string
        coutFile = 0
        for singleFile in fileToScan:
            SignificantString = []
            text = ""
            with open(f'data/{singleFile}') as file:
                for line in file:
                    text = line
            strings = text.split(". ")
            # print(len(strings))
            for string in strings:
                for word in TopWords:
                    if len(word) != 0 and word[0] in string:
                        SignificantString.append(string)

            # remove duplicates strings from SignificantString
            seen = set()
            results = []
            for item in SignificantString:
                if item not in seen:
                    seen.add(item)
                    results.append(item)
            SignificantString = results

            # write SignificantString to file
            with open(f'riassunto/{year}_SignificantString_{coutFile}.doc', 'w') as file:
                for string in SignificantString:
                    file.write("- ")
                    file.write(string)
                    file.write("\n\n\n")

            coutFile += 1
