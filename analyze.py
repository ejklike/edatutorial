from collections import Counter
from itertools import chain

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

import ast

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression



stop_list = ['기자', '저작권', '본문', '기사', '뉴스', '제공', '전재', '배포', '때문',
             # 이재명
             '이재명', '대표', '재명이', '민주당', '더불어민주당',
             # 윤석열
             '윤석열', '대통령', '국힘', '국민의힘', '국민']

synonym_dict = dict()


def get_y(cand):
    return np.array([1 if label==cand else 0 for label in df['label']])


def printf_list(num_feature, feature_score, feature_name):
    top_k_indices = np.argsort(-feature_score)[:num_feature]
    for i, k in enumerate(top_k_indices):
        print('{}'.format(feature_name[k]), end=', ')
        if (i+1)%10 == 0:
            print('')


if __name__ == '__main__':

    MIN_COUNT = 30

    # data loading
    df = pd.read_csv('./data/df_all_preprocessed_noun.csv', sep=',', quoting=1)
    df['nouns'] = df['nouns'].apply(lambda x: ast.literal_eval(x))
    print("Length of df: ", len(df))

    
    # input features
    news_noun_list = df['nouns'].tolist()
    all_nouns = chain(*news_noun_list)
    word_counts = sorted(Counter(all_nouns).items(), 
                         key=lambda x: x[1], 
                         reverse=True)
    feature_list = [
        word for word, count in word_counts
        if count > MIN_COUNT and len(word) > 1 and word not in stop_list
    ]
    print('Input feature list length: ', len(feature_list))

    # tf array
    news_count_list = [Counter(news_noun) for news_noun in news_noun_list]
    tf = pd.DataFrame(
        [[count[f] > 0 for f in feature_list] for count in news_count_list],
        columns=feature_list)
    
    tf = tf.astype(np.float32)
    print('TF array shape:', tf.shape)
    
    # synonym preprocessing
    for k, v in synonym_dict.items():
        if k in feature_list and v in feature_list:
            print('merging feature {} to {}'.format(k, v))
            tf[v] = tf[v] + tf[k]
            del tf[k]

    print('tf shape:', tf.shape)

    # add press tokens
    press_df = df['press'].value_counts().sort_values(ascending=False)
    press_df = press_df[press_df > MIN_COUNT]
    press_dict = dict()
    for p in press_df.index:
        feature = 'p_' + p
        if p not in feature_list:
            press_dict[feature] = df['press'].apply(lambda x: 1 if x == p else 0)
        else:
            press_dict[feature] = tf[p] + df['press'].apply(lambda x: 1 if x == p else 0)
            press_dict[feature] = press_dict[feature].apply(lambda x: min(1, x)).astype(np.float32)
            del tf[p]
            feature_list.remove(p)
    tf = pd.concat([tf, pd.DataFrame(press_dict)], axis=1)
    feature_list = list(tf.columns)
    print('after adding press tokens...')
    print('Feature list length: ', len(feature_list))
    print('tf shape:', tf.shape)

    # tf-idf array
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(tf.values)
    
    # dataset
    cand = '이재명' # positive class
    X = tfidf.toarray()
    y = get_y(cand)

    # train-test split
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)
    print('train-test split done')
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    
    # logistic regression
    clf = LogisticRegression(penalty='l2', C=10.0, solver='liblinear')
    clf.fit(X_train, y_train)
    
    y_train_score = clf.predict_proba(X_train)[:, 1]
    yhat_train = clf.predict(X_train)
    print('Train data results:')
    print(f'Accuracy : {accuracy_score(y_train, yhat_train)}')
    print(f'AUC score: {roc_auc_score(y_train, y_train_score)}')

    y_test_score = clf.predict_proba(X_test)[:, 1]
    yhat_test = clf.predict(X_test)
    print('Test data results:')
    print(f'Accuracy : {accuracy_score(y_test, yhat_test)}')
    print(f'AUC score: {roc_auc_score(y_test, y_test_score)}')
    
    coeff_values = clf.coef_[0]    
    sorted_words = sorted(zip(feature_list, coeff_values), key=lambda x: x[1])
    
    n_words = 50
    print('--------이재명--------')
    printf_list(n_words, coeff_values, feature_list)
    print('--------윤석열--------')
    printf_list(n_words, -coeff_values, feature_list)
