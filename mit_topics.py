# Import Module
import os
import text_preprocessing as tp

if __name__ == "__main__":
    # Working Folder
    os.chdir("data")
    count = 0
    for file in os.listdir():
        count = count + 1

        if file.endswith(".txt"):
            input_file = open(file, encoding="utf8")
            file_text = input_file.read()

            file_text = tp.remove_whitespace(file_text)
            file_text = tp.tokenization(file_text)
            file_text = tp.stopword_removing(file_text)
            file_text = tp.pos_tagging(file_text)
            file_text = tp.lemmatization(file_text)

            print(tp.word_count(file_text))
            tp.tag_cloud(file_text)

        if(count == 10):
            break