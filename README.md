# MITopics

MITopics is a Python project aimed at finding the main topics from a set of 30 years' worth of management articles from the MIT Sloan Review. The project utilizes various techniques to identify topics, including LDA, LSA, Top2Vec, and two other techniques based on word embedding and clustering. The analysis of the documents was performed on both the collection of articles for each year and decade, and the techniques used were compared. A timeline was generated to see how the topic of management evolved over the years. The results can be viewed in the PDF file "MITopics.pdf."

## Installation

To install MITopic, first clone the repository:

`git clone https://github.com/mgiorgi13/MITopics.git`

Then, navigate to the project directory and install the required dependencies:

`pip install -r requirements.txt`

## Usage

To use MITopic, run the following command:

`python MITopic.py`

This will run the script and let you apply different techniques to the documents, showing you the results as list of topics or barchart or wordcloud.

## Result examples

![topic list](https://github.com/mgiorgi13/MITopics/blob/main/result_example/topic%20list.png)
![bar chart](https://github.com/mgiorgi13/MITopics/blob/main/result_example/bar%20chart.png)
![tag cloud](https://github.com/mgiorgi13/MITopics/blob/main/result_example/wordcloud.png)
