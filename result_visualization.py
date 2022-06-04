import matplotlib.pyplot as plt
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


def tag_cloud(words):
    # Start with one review:
    text = " ".join([str(item) for item in words])

    # Create and generate a word cloud image:
    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
