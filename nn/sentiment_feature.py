import os
import sys
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer #need for sentiment feature

directory_path = './features/'
headline_out = '_headlines_sentiment_feature'
body_out = '_bodies_sentiment_feature'

def sentiment_feature_generate(name, headlines, bodies):
    """
    generate sentiment feature by given dataset

    name: name of data set, ex) train, competition ...
    headlines: array contains titles12
    bodies: array contains bodies +) heads and bodies most be paired.
    """
    if len(headlines) != len(bodies):
        print("Check headlines size and bodies size for sentiment_feature_generate")
        sys.exit(1)

    sid = SentimentIntensityAnalyzer()
    print('generating sentiment feature for headlines')
    head_sentiment = list(sid.polarity_scores(headlines[0]).values())
    heads_sentiment_feature = [head_sentiment]
    for idx in range(1, len(headlines)):
        head_sentiment = list(sid.polarity_scores(headlines[idx]).values())
        heads_sentiment_feature = np.vstack((heads_sentiment_feature, head_sentiment))
    np.save(directory_path + name + headline_out, heads_sentiment_feature)

    print('generating sentiment feature for bodies')
    body_sentiment = list(sid.polarity_scores(bodies[0]).values())
    bodies_sentiment_feature = [body_sentiment]
    for idx in range(1, len(bodies)):
        body_sentiment = list(sid.polarity_scores(bodies[idx]).values())
        bodies_sentiment_feature = np.vstack((bodies_sentiment_feature, body_sentiment))
    np.save(directory_path + name + body_out, bodies_sentiment_feature)

    return np.hstack((heads_sentiment_feature, bodies_sentiment_feature))

def sentiment_feature_read(name):
    """
    read pre-generated sentiment feature

    name: name of data set, ex) train, competition ...
    """
    if not check_sentiment_feature_exist(name):
        print(name + " doesn't existt for sentiment")
        sys.exit(1)

    head_sentiment = np.load(directory_path + name + headline_out + '.npy')
    body_sentiment = np.load(directory_path + name + body_out + '.npy')
    
    if len(head_sentiment) != len(body_sentiment):
        print("Check headlines size and bodies size for sentiment_feature_read")

    return np.hstack((head_sentiment, body_sentiment))

def check_sentiment_feature_exist(name):
    return os.path.exists(directory_path + name + headline_out + '.npy') and os.path.exists(directory_path + name + body_out + '.npy')

if __name__ == '__main__':
    from utils.dataset import DataSet

    d = DataSet(name = 'train', path = './')
    train_heads, train_bodies = d.get_headlines_bodies()
    sentiment_feature_generate('train', train_heads, train_bodies)
    sentiment_x = sentiment_feature_read('train')
    print(sentiment_x[1])
