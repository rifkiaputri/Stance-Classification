import os
import re
import sys
import nltk
""" need
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
"""
import numpy as np

#directory_path = './nlp_project/overlapping_feature/'
overlapping_out = '_overlapping_feature.out'

_wnl = nltk.WordNetLemmatizer()

def normalize_word(w):
    return _wnl.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def overlapping_feature_generate(name, headlines, bodies):
    """
    generate overlapping feature by given dataset

    name: name of data set, ex) train, competition ...
    headlines: array contains titles12
    bodies: array contains bodies +) heads and bodies most be paired.
    """
    if len(headlines) != len(bodies):
        print("Check headlines size and bodies size for overlapping_feature_generate")
        sys.exit(1)

    print('generating overlapping feature')
    X = []
    for headline, body in zip(headlines, bodies):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        feature = [len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(feature)
    np.savetxt('./' + name + overlapping_out, X)

    return X

def overlapping_feature_read(name):
    """
    read pre-generated overlapping feature

    name: name of data set, ex) train, competition ...
    """
    if not check_overlapping_feature_exist(name):
        print(name + " doesn't exists")
        sys.exit(1)

    overlapping = np.genfromtxt('./' + name + overlapping_out)

    return overlapping

def check_overlapping_feature_exist(name):
    if not os.path.exists("./" + name + overlapping_out):
        return False
    return True

if __name__ == '__main__':
    from utils.dataset import DataSet

    d = DataSet(name = 'train', path = './')
    train_heads, train_bodies = d.get_headlines_bodies()
    #overlapping_feature_generate('train', train_heads, train_bodies)
    overlapping_x = overlapping_feature_read('train')
