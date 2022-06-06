import os
from wordcloud import WordCloud
import pandas as pd
from matplotlib import pyplot as plt

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



# histogram of the 50th words with frequency
def histogram(title, data):

    wordlist = []
    for i in range(0, len(data)):
        wordlist.append(data[i])

    list = word_count(wordlist)

    words = []
    frequency = []

    for elem in list[:50]:
        if elem[1] >= 10:
            words.append(elem[0])
            frequency.append(elem[1])

    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Horizontal Bar Plot
    ax.barh(words, frequency)

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='grey')

    # Add Plot Title
    ax.set_title(title,
                 loc='left', )

    # Show Plot
    if not os.path.exists(f"output/{title}/wordcloud"):
        os.makedirs(f"output/{title}/wordcloud")
    plt.savefig(f"output/{title}/wordcloud/histogram_{title}.png")
