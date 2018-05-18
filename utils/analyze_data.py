import nltk
from dataset import DataSet

nltk.download('punkt')
train = DataSet("train", "fnc-1")
tokenized_articles = dict()
max_l = 0
for body_id, article in train.articles.items():
    tokenized_articles[body_id] = nltk.word_tokenize(article)
    len_words = len(tokenized_articles[body_id])
    if len_words > max_l:
        max_l = len_words

print('Maximum length of the body: ', max_l)