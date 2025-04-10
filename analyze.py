from collections import Counter
from itertools import chain

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

import ast

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression

from konlpy.tag import Okt
from kiwipiepy import Kiwi


# 각 프로세스에서 개별적으로 Kiwi 인스턴스 생성
def extract_nouns_batch(text_batch):
    kiwi = Kiwi()  # 프로세스마다 Kiwi 생성 (초기화 1회)
    tokens_batch = [kiwi.tokenize(text, normalize_coda=False, saisiot=False) 
                    for text in text_batch]
    return [[x.form for x in tokens if x.tag[0] == 'N'] for tokens in tokens_batch]


synonym_dict = {
    '국민의힘': '국힘',
    '더불어민주당': '민주당',
    '이재명대표': '이재명',
    # '국민의 힘': '국힘',
    # '더불어 민주당': '민주당',
    # '이재명 대표': '이재명',
}

stop_list = ['기자', '저작권', '본문', '기사', '뉴스', '제공',
             # 이재명
             '이재명', '대표', '재명이', '민주당', '더불어민주당',
             # 윤석열
             '윤석열', '대통령', '국힘', '국민의힘']


def get_df_noun_extracted():
    # data loading
    df1 = pd.read_csv('./data/lee_all.csv', sep=',', quoting=1)
    df1['label'] = '이재명'
    df2 = pd.read_csv('./data/yoon_all.csv', sep=',', quoting=1)
    df2['label'] = '윤석열'
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    
    # remove duplicates
    df = df.drop_duplicates('url', keep='first')
    
    # remove nan news
    df = df.dropna(subset='text')
    
    # remove short news
    df = df[df.text.str.len() > 100]

    # remove news not relevant to candidates
    df = df[df.text.str.contains('윤석열') | df.text.str.contains('이재명')]

    # # preprocess synonyms
    # for i, row in df.iterrows():
    #     text = row.text
    #     for k, v in synonym_dict.items():
    #         text = text.replace(k, v)
    #     df.loc[i, 'text'] = text
    
    # noun extraction --- Okt
    # okt = Okt()
    # news_noun_list = []
    # for c in df['text']:
    #     news_noun_list.append(okt.nouns(c))

    # noun extraction --- Kiwi
    # (1) 병렬 미사용
    # news_noun_list = []
    # kiwi = Kiwi()
    # for c in tqdm(df['text']):
    #     tokens = kiwi.tokenize(text, normalize_coda=False, saisiot=False)
    #     nouns = [x.form for x in tokens if x.tag[0] == 'N']
    #     news_noun_list.append(nouns)

    # (2) 병렬 처리
    # news_noun_list = extract_nouns_batch_process(df['text'].tolist())
    df_texts = df['text'].tolist()
    BATCH_SIZE = min(1000, len(df_texts) // os.cpu_count() + 1)

    # 입력 데이터를 배치 단위로 나누기
    batches = [df_texts[i:i + BATCH_SIZE] for i in range(0, len(df_texts), BATCH_SIZE)]

    # 병렬 처리
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(extract_nouns_batch, batches), total=len(batches)))

    # 결과 병합
    news_noun_list = [item for sublist in results for item in sublist]

    # df 에 noun_list 추가
    df['nouns'] = news_noun_list
    df.to_csv('./data/df_all_preprocessed_noun.csv', sep=',', quoting=1, index=False)

    return df


def get_y(cand):
    return np.array([1 if label==cand else 0 for label in df['label']])


def printf_list(num_feature, feature_score, feature_name):
    top_k_indices = np.argsort(-feature_score)[:num_feature]
    for i, k in enumerate(top_k_indices):
        print('{}'.format(feature_name[k]), end=', ')
        if (i+1)%10 == 0:
            print('')


if __name__ == '__main__':

    # data loading
    if os.path.exists('./data/df_all_preprocessed_noun.csv'):
        df = pd.read_csv('./data/df_all_preprocessed_noun.csv', sep=',', quoting=1)
        df['nouns'] = df['nouns'].apply(lambda x: ast.literal_eval(x))
    else:
        df = get_df_noun_extracted()
    print("Length of df: ", len(df))

    news_noun_list = df['nouns'].tolist()
    print('len(news_noun_list)', len(news_noun_list))

    # input features
    all_nouns = chain(*news_noun_list)
    word_counts = sorted(Counter(all_nouns).items(), key=lambda x: x[1], reverse=True)
    # print(word_counts[:30])
    feature_list = [
        word for word, count in word_counts
        if count > 30 and len(word) > 1 and word not in stop_list
    ]
    print('Feature list length: ', len(feature_list))

    # tf array
    news_count_list = [Counter(news_noun) for news_noun in news_noun_list]
    tf = pd.DataFrame(
        [[count[f] > 0 for f in feature_list] for count in news_count_list],
        columns=feature_list)
    
    tf = tf.astype(np.float32)
    print(tf.shape)
    
    # synonym preprocessing
    for k, v in synonym_dict.items():
        if k in feature_list and v in feature_list:
            print('merging {} and {}'.format(k, v))
            tf[v] = tf[v] + tf[k]
            del tf[k]

    print(tf.shape)

    # add press tokens
    press_df = df['press'].value_counts().sort_values(ascending=False)
    press_df = press_df[press_df > 30]
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
    print('Feature list length: ', len(feature_list))

    print(tf.shape)

    # tf-idf array
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(tf.values)
    X = tfidf.toarray()
    
    # logistic regression
    cand = '이재명' # positive class
    y = get_y(cand)
    clf = LogisticRegression(penalty='l1', C=10.0, solver='liblinear')
    clf.fit(X, y)
    
    y_score = clf.predict_proba(X)[:, 1]
    yhat = clf.predict(X)
    print(f'Accuracy : {accuracy_score(y, yhat)}')
    print(f'AUC score: {roc_auc_score(y, y_score)}')
    
    coeff_values = clf.coef_[0]    
    sorted_words = sorted(zip(feature_list, coeff_values), key=lambda x: x[1])
    
    n_words = 50
    print('--------이재명--------')
    printf_list(n_words, coeff_values, feature_list)
    print('--------윤석열--------')
    printf_list(n_words, -coeff_values, feature_list)
