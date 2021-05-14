import os
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import svm
import spacy
import math
import time

# python -m spacy download en
sp = spacy.load('en_core_web_sm')
sp_stopwords = sp.Defaults.stop_words

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

RW0 = "RW0"
RWeidf = "RWeidf"
RWeoc = "RWeoc"

C_value = 0.95
D = 0.85
MIN_DIFF = 1e-5

MAX_FEATURES = 2000
RW_MODE = RWeidf
WINDOW_SIZE = 4
STEPS = 2


def remove_stop_words(data):
    words = sp(str(data))
    new_text = ""
    for w in words:
        if w not in sp_stopwords and len(w) > 1:
            new_text = new_text + " " + str(w)
    return new_text


def clean_data(data):

    # remove_punctuation
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')

    # remove_apostrophe
    data = np.char.replace(data, "'", "")

    result = " "
    # remove_integers
    result = ''.join([i for i in str(data) if not i.isdigit()])

    return str(result)


def preProccesing(dir, mode):

    # Read the file names
    emails_dir = os.listdir(dir)
    emails_dir.sort()
    N = len(emails_dir)
    # Array to hold all the words in the emails
    emails = [0] * N
    # Collecting all words from those emails
    for i in range(len(emails_dir)):
        print(mode + " pre proccesing :", round((i/N) * 100, 2), "%")

        content = []
        m = open(os.path.join(dir, emails_dir[i]))

        for line_index, line in enumerate(m):
            if line_index == 2:
                data = clean_data(line)
                data = remove_stop_words(data)
                # data = line

        emails[i] = data

    return emails


def occurence_features(documents, dictionary):

    features_matrix = np.zeros((len(documents), len(dictionary)))

    # collecting the number of occurances of each of the words in the emails
    for doc_index, document in enumerate(documents):
        words = document.split()
        for word_index, word in enumerate(dictionary):
            features_matrix[doc_index,
                            word_index] = words.count(word)

    return features_matrix


def Dictionary_and_TF_IDF_feature(documents):

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    # vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    tokens = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()

    return tokens, denselist


def TF_IDF_feature(documents, dictionary):

    # tf_matrix = np.zeros((len(documents), len(dictionary)))
    # idf_matrix = np.zeros((len(documents), len(dictionary)))
    tf_idf_matrix = np.zeros((len(documents), len(dictionary)))

    N = len(documents)
    size = len(dictionary)

    for word_index, word in enumerate(dictionary):
        print("test data TF-IDF :", round((word_index/size) * 100, 2), "%")

        df = 0
        occurance = 0
        for doc_index, document in enumerate(documents):
            words = document.split()
            occurance = words.count(word)
            # if occurance != 0:
            #     df += 1

            tf = occurance/len(words)
            idf = math.log(N/(df+1))

            # tf_matrix[doc_index, word_index] = tf
            # tf_matrix[doc_index, word_index] = idf
            tf_idf_matrix[doc_index, word_index] = tf * idf

    return tf_idf_matrix


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


def get_token_pairs(window_size, document):
    """Build token_pairs from windows in sentences"""
    token_pairs = list()

    words = document.split()
    for i, word in enumerate(words):
        for j in range(i+1, i + window_size):
            if j >= len(words):
                break
            pair = (word, words[j])
            if pair not in token_pairs:
                token_pairs.append(pair)

    return token_pairs


def get_matrix(vocab, token_pairs, d_matrix, C):
    """Get normalized matrix"""

    # Build matrix
    vocab_size = len(vocab)
    g = np.zeros((vocab_size, vocab_size), dtype='float')
    for word1, word2 in token_pairs:
        if word1 in vocab and word2 in vocab:
            i, j = vocab.index(word1), vocab.index(word2)
            g[i][j] = 1

    # Get Symmeric matrix
    g = symmetrize(g)

    np.multiply(g, d_matrix)

    # all elements * C
    g * C

    # Normalize matrix by column
    norm = np.sum(g, axis=0)

    # # this is ignore the 0 element in norm
    g_norm = np.divide(g, norm, where=norm != 0)

    return g_norm


def rw_feature(documents, dictionary, E, RWmode):
    rw_matrix = np.zeros((len(documents), len(dictionary)))
    N = len(dictionary)
    size = len(documents)

    if RWmode == RW0:
        d_matrix = np.full((N, N), 1)
        C = 1

    for doc_index, document in enumerate(documents):
        print(RWmode + " featureing :", round((doc_index/size) * 100, 2), "%")

        # for RWeidf and RWeoc
        if RWmode != RW0:
            C = C_value
            E_vec = np.array([E[doc_index]])
            d_matrix = E_vec.transpose() * E_vec

            if RWmode == RWeidf:
                d_matrix / d_matrix.max()

        # Get token_pairs from windows
        token_pairs = get_token_pairs(WINDOW_SIZE, document)

        # Get normalized matrix
        g = get_matrix(dictionary, token_pairs, d_matrix, C)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(dictionary))

        # Iteration
        previous_pr = 0
        for epoch in range(STEPS):
            pr = (1-D) + D * np.dot(g, pr)

            if abs(previous_pr - sum(pr)) < MIN_DIFF:
                break
            else:
                previous_pr = sum(pr)

        # add value to all elements of pr
        if RWmode == RW0:
            pr = pr*D + (1-D)/N

        # for RWeidf and RWeoc
        else:
            pr + (1-D)/N

        rw_matrix[doc_index] = pr

    return rw_matrix


def build_labels(dir):
    # Read the file names
    emails = os.listdir(dir)
    emails.sort()
    # ndarray of labels
    labels_matrix = np.zeros(len(emails))

    for index, email in enumerate(emails):
        if re.search('spms*', email):
            labels_matrix[index] = 1
        else:
            labels_matrix[index] = 0

        # labels_matrix[index] = 1 if re.search('spms*', email) else 0

    return labels_matrix


def TF_features(documents, dictionary):

    features_matrix = np.zeros((len(documents), len(dictionary)))

    # collecting the number of occurances of each of the words in the emails
    for doc_index, document in enumerate(documents):
        words = document.split()
        for word_index, word in enumerate(dictionary):
            features_matrix[doc_index,
                            word_index] = words.count(word)

        # normalize
        temp = features_matrix[doc_index]
        norm = np.linalg.norm(temp)
        features_matrix[doc_index] = temp/norm

    return features_matrix


start = time.time()

train_dir = './LSpam/train_data'

print('1. Train data pre process')
train_data = preProccesing(train_dir, "train")

print('2. Create Dictionary and Building training features and labels')
dictionary, tfidf = Dictionary_and_TF_IDF_feature(train_data)
# features_train = TF_features(train_data, dictionary)

features_train = rw_feature(train_data, dictionary, tfidf, RW_MODE)

labels_train = build_labels(train_dir)

nb_classifier = MultinomialNB()
svm_classifier = svm.LinearSVC()

print('4. Test data pre process')
test_dir = './LSpam/test_data'
test_data = preProccesing(test_dir, "test")


print('5. Building the test features and labels')
# features_test = TF_features(test_data, dictionary)
tfidf = TF_IDF_feature(test_data, dictionary)
features_test = rw_feature(test_data, dictionary, tfidf, RW_MODE)

labels_test = build_labels(test_dir)

print('6. Training the classifier')
nb_classifier.fit(features_train, labels_train)
svm_classifier.fit(features_train, labels_train)


print('7. Calculating accuracy of the trained classifier')
nb_accuracy = nb_classifier.score(features_test, labels_test)
svm_accuracy = svm_classifier.score(features_test, labels_test)

duration = round(time.time() - start, 4)

print("\nMax Features:", MAX_FEATURES,
      "  Window size:", WINDOW_SIZE, "  Steps:", STEPS)

print('\nSVM Accuracy : ', svm_accuracy)
print('NB Accuracy : ', nb_accuracy)

print('execute time:', duration)
