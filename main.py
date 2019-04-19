"""
TTCS - Technion Text Column Summarization.

Supervisor:
    - Professor Benny Kimelfeld
        Email: bennykcs.technion.ac.il

Developers:
    - Omar Jaml
        Email: omar.jaml.96@gmail.com
    - Samah Anabusy
        Email: mousy.on@gmail.com
"""

from stopword import clear_stopwords
from nltk.corpus import  wordnet
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def getmax(text_l, syns):
    counter_s = 0
    max_str = ""
    for w in syns:
        if text_l.count(w) > counter_s:
            max_str = w
            counter_s = text_l.count(w)
    return max_str


def main():

    print("Please enter the name of the Dataset:\n")
    dataset_name = input()
    dsf = pd.read_csv(dataset_name)

    print("Please enter the name of the text column you wish to summarize:\n")
    column_name = input()
    column_text = dsf[column_name]

    print("Please enter the language of the Dataset:\n")
    language = input()

    filtered_text = clear_stopwords(column_text, language)

    text = " ".join(w for w in filtered_text)

    wordcloud = WordCloud(max_words=100, stopwords=filtered_text, background_color="white", colormap="gist_rainbow").generate(text)


    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    synonyms = []

    for word in filtered_text:
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())

        new_syn = intersection(filtered_text, synonyms)
        max_s = getmax(filtered_text, new_syn)
        for w in new_syn:
            if w != max_s:
                while w in filtered_text:
                    filtered_text.remove(w)
                    filtered_text.append(max_s)


    text = " ".join(w for w in filtered_text)

    wordcloud = WordCloud(max_words=100, collocations=False, stopwords=filtered_text, background_color="white", colormap="gist_rainbow").generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()




