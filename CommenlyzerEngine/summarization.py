from gensim.summarization import summarize


def text_summarize(text, word_count=100):
    return summarize(text, word_count=word_count)
