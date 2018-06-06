import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

directory_path = './features/tfidf/'
tfidf_out = '_tfidf_feature'
feature_size = 5000
batch_size = 500

stop_words = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]

def tfidfVectorizer_init(train_heads, train_bodies, test_heads = None, test_bodies = None):
    if(test_heads != None or test_bodies != None):
        return TfidfVectorizer(max_features=feature_size, stop_words=stop_words).fit(train_heads + train_bodies + test_heads + test_bodies)
    else:
        return TfidfVectorizer(max_features=feature_size, stop_words=stop_words).fit(train_heads + train_bodies)

def tfidf_feature_generate(name, tfidf_vectorizer, train_x):
    def get_vector(one_x):
        head_vec = tfidf_vectorizer.transform([one_x['headline']]).toarray()
        body_vec = tfidf_vectorizer.transform([one_x['article']]).toarray()
        cos_sim = cosine_similarity(head_vec, body_vec)[0].reshape(1, 1)
        return head_vec[0], cos_sim[0], body_vec[0]

    head, sim, body = get_vector(train_x[0])
    feature = np.concatenate((head, sim, body))

    for idx in range(1, len(train_x)):
        if idx % batch_size == 0:
            print(str(idx) + ':' + str(len(train_x)))
            np.save(directory_path + name + tfidf_out + '_' + str(idx), feature)
            head, sim, body = get_vector(train_x[idx])
            feature = np.concatenate((head, sim, body))
            continue
        head, sim, body = get_vector(train_x[idx])
        feature = np.vstack((feature, np.concatenate((head, sim, body))))

    np.save(directory_path + name + tfidf_out + '_last', feature)

    return 1

def tfidf_feature_read(name):
    if not check_tfidf_feature_exist(name):
        print(name + " doesn't exist for tfidf")
        sys.exit(1)

    tfidf_feature = np.load(directory_path + name + tfidf_out + '_' + str(batch_size) + '.npy')

    for idx in range(1, 100):
        if not check_tfidf_feature_exist(name, (idx + 1) * batch_size):
            data = np.load(directory_path + name + tfidf_out + '_last' + '.npy')
            tfidf_feature = np.vstack((tfidf_feature, data))
            break
        data = np.load(directory_path + name + tfidf_out + '_' + str((idx + 1) * batch_size) + '.npy')
        tfidf_feature = np.vstack((tfidf_feature, data))

    return tfidf_feature

def check_tfidf_feature_exist(name, num = 500):
    return os.path.exists(directory_path + name + tfidf_out + '_' + str(num) + '.npy')
        
