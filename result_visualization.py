import os
from wordcloud import WordCloud

def word_count(str):
    counts = dict()
    words = str

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    sort_orders = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    return sort_orders


def tag_cloud(words, year, path):
    # Start with one review:
    text = " ".join([str(item) for item in words])

    # Create and generate a word cloud image:
    wordcloud = WordCloud(width=1920, height=1080,  background_color="white").generate(text)

    if path != "-":
        if not os.path.exists(path):
            os.makedirs(path)
        wordcloud.to_file(f"{path}/wordcloud.png")
    else:
        if not os.path.exists(f"output/{year}/wordcloud"):
            os.makedirs(f"output/{year}/wordcloud")
        wordcloud.to_file(f"output/{year}/wordcloud/wordcloud.png")



