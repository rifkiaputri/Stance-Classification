import os
import sys
import numpy as np
import clean_string

directory_path = './features/'
overlapping_out = '_overlapping_feature'

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

    feature = []
    for headline, body in zip(headlines, bodies):
        clean_headline = clean_string.clean(headline)
        clean_body = clean_string.clean(body)
        clean_headline = clean_string.get_tokenized_lemmas(clean_headline)
        clean_body = clean_string.get_tokenized_lemmas(clean_body)
        temp = [len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        feature.append(temp)
    np.save(directory_path + name + overlapping_out, feature)

    return feature

def overlapping_feature_read(name):
    """
    read pre-generated overlapping feature

    name: name of data set, ex) train, competition ...
    """
    if not check_overlapping_feature_exist(name):
        print(name + " doesn't exist for overlapping")
        sys.exit(1)

    return np.load(directory_path + name + overlapping_out + '.npy')

def check_overlapping_feature_exist(name):
    return os.path.exists(directory_path + name + overlapping_out + '.npy')

if __name__ == '__main__':
    from utils.dataset import DataSet

    d = DataSet(name = 'train', path = './')
    train_heads, train_bodies = d.get_headlines_bodies()
    overlapping_feature_generate('train', train_heads, train_bodies)
    overlapping_x = overlapping_feature_read('train')
    print(overlapping_x)
