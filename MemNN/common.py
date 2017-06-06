def split_line(line):
    """
    :param line: from an input data file.
    :return: lower-cased words split by whitespaces.
    """
    return line.split('\t')[3].strip().lower().split(' ')


def clean_words(words):
    """
    :param words: a list of raw words.
    :return: a list of words where each word is cleaned from special symbols.
    """
    for w in words:
        w = w.strip('".\'?)(:,!\\[]=/')
        if w.endswith('\'s'):
            w = w[:len(w)-2]
        if w is not '':
            yield w
