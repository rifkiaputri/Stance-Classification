import os
import re
import nltk
import sys
import numpy as np
import clean_string

directory_path = './features/'
count_out = '_count_feature'

_refuting_words = ['fake', 'fraud', 'hoax', 'false', 'deny', 'denies', 'not', 'despite', 'nope', 'doubt', 'doubts', 'bogus', 'debunk', 'pranks', 'retract']

def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(clean_string.remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features

def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features

def count_feature_generate(name, headlines, bodies):
    """
    generate count feature by given dataset

    name: name of data set, ex) train, competition ...
    headlines: array contains titles12
    bodies: array contains bodies +) heads and bodies most be paired.
    """
    if len(headlines) != len(bodies):
        print("Check headlines size and bodies size for count_feature_generate")
        sys.exit(1)

    def calculate_polarity(text):
        tokens = clean_string.get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean_string.clean(headline).split(" "):
            if headline_token in clean_string.clean(body):
                bin_count += 1
            if headline_token in clean_string.clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean_string.remove_stopwords(clean_string.clean(headline).split(" ")):
            if headline_token in clean_string.clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean_string.clean(body)
        clean_headline = clean_string.clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    feature = []
    for headline, body in zip(headlines, bodies):
        clean_headline = clean_string.clean(headline)
        clean_body = clean_string.clean(body)

        grams_feat = binary_co_occurence(headline, body) + binary_co_occurence_stops(headline, body) + count_grams(headline, body)
        polarity_h_feat = [calculate_polarity(clean_headline)]
        polarity_b_feat = [calculate_polarity(clean_body)]

        clean_headline = clean_string.get_tokenized_lemmas(clean_headline)
        clean_body = clean_string.get_tokenized_lemmas(clean_body)
        
        refute_feat = [1 if word in clean_headline else 0 for word in _refuting_words]
        overlapping_feat = [len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]

        temp = grams_feat + refute_feat + polarity_h_feat + polarity_b_feat + overlapping_feat
        feature.append(temp)
    np.save(directory_path + name + count_out, feature)

    return np.asarray(feature)

def count_feature_read(name):
    """
    read pre-generated count feature

    name: name of data set, ex) train, competition ...
    """
    if not check_count_feature_exist(name):
        print(name + " doesn't exist for count")
        sys.exit(1)

    return np.load(directory_path + name + count_out + '.npy')

def check_count_feature_exist(name):
    return os.path.exists(directory_path + name + count_out + '.npy')

if __name__ == '__main__':
    from utils.dataset import DataSet

    d = DataSet(name = 'competition_test', path = './')
    train_heads, train_bodies = d.get_headlines_bodies()
    count_feature_generate('competition_test', train_heads, train_bodies)
    count_x = count_feature_read('competition_test')
    print(count_x)
