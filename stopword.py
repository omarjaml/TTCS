from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def clear_stopwords(text_column, language):
    """
    This function clearing the text column from stop words

    :param text_column: column containing the text review, etc..
    :return: cleaned text list
    """

    text = " ".join(r for r in text_column)
    text = text.lower()

    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(text)

    filtered_sentence = [w for w in word_tokens if w not in stop_words]

    return filtered_sentence
