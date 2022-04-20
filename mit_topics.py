# Import Module
import os
import text_preprocessing as tp

# Working Folder
os.chdir("data")

for file in os.listdir():
    if file.endswith(".txt"):
        input_file = open(file, encoding="utf8")
        file_text = input_file.read()

        print(file_text)