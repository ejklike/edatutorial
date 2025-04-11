import pandas as pd

import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from kiwipiepy import Kiwi

def extract_nouns_batch(text_batch):
    kiwi = Kiwi()  # 프로세스마다 Kiwi 생성 (초기화 1회)
    tokens_batch = [kiwi.tokenize(text, normalize_coda=False, saisiot=False) 
                    for text in text_batch]
    return [[x.form for x in tokens if x.tag[0] == 'N'] for tokens in tokens_batch]


if __name__ == '__main__':
    
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

    df_texts = df['text'].to_list()


    # noun extraction --- Okt
    # okt = Okt()
    # news_noun_list = []
    # for c in df_texts:
    #     news_noun_list.append(okt.nouns(c))

    # noun extraction --- Kiwi
    # (1) 병렬 미사용
    # news_noun_list = []
    # kiwi = Kiwi()
    # for c in tqdm(df_texts):
    #     tokens = kiwi.tokenize(text, normalize_coda=False, saisiot=False)
    #     nouns = [x.form for x in tokens if x.tag[0] == 'N']
    #     news_noun_list.append(nouns)

    # (2) 병렬 처리
    n_core = os.cpu_count()
    BATCH_SIZE = len(df_texts) // n_core
    
    # 입력 데이터를 배치 단위로 나누기
    batches = [df_texts[i:i + BATCH_SIZE] 
               for i in range(0, len(df_texts), BATCH_SIZE)]
    
    # 배치 단위 병렬 처리
    with ProcessPoolExecutor(max_workers=n_core) as executor:
        results = list(tqdm(
            executor.map(extract_nouns_batch, batches), 
            total=len(batches)))

    # 결과 병합
    news_noun_list = [item for sublist in results for item in sublist]

    # df 에 noun_list 추가
    df['nouns'] = news_noun_list
    df.to_csv('./data/df_all_preprocessed_noun.csv', sep=',', quoting=1, index=False)






