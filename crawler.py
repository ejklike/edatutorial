import os
import sys
import datetime
import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from newspaper import Article, ArticleException

import pandas as pd


def validate(date_text):
    try:
        datetime.date.fromisoformat(date_text)
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")


def get_news_title_text_date(url):
    # 언어를 한국어로 설정하고 URL을 전달해 Article 클래스의 객체 생성
    article = Article(url, language='ko')
    # 지정된 웹 페이지를 다운로드
    article.download()
    # 다운로드한 웹 페이지를 분석하고 필요한 정보를 추출
    article.parse()
    return article.title, article.text, article.publish_date


def scroll_down(driver, n_scroll=3, sleep_time=2, verbose=True):
    #스크롤 내리기 이동 전 위치
    old_location = driver.execute_script(
        "return document.body.scrollHeight")
    actions = driver.find_element('css selector', 'body')

    if verbose: print('scroll down.. ', end='')
    for scroll_count in range(n_scroll):
        actions.send_keys(Keys.END)
        time.sleep(sleep_time)
        if verbose: print(scroll_count+1, end=' ')
        #늘어난 스크롤 높이
        new_location = driver.execute_script(
            "return document.body.scrollHeight")
        #늘어난 스크롤 위치와 이동 전 위치 같으면(더 이상 스크롤이 늘어나지 않으면) 종료
        if new_location == old_location:
            if verbose: print(f'scroll limit reached..')
            break
        #스크롤 위치값을 수정
        old_location = new_location


def get_news_list(keyword, ds, de, n_scroll=3, 
                  headless=True, verbose=True):

    # url 정의
    url = f'https://search.naver.com/search.naver?where=news&query={keyword}&pd=3&ds={ds}&de={de}'

    # 옵션 생성
    options = webdriver.ChromeOptions()
    if headless:
        # 창 숨기는 옵션 추가
        options.add_argument("headless")

    # driver 정의
    driver = webdriver.Chrome(options=options)

    # url 접속
    driver.get(url)
    driver.implicitly_wait(2) # 2초 안에 웹페이지를 load하면 바로 넘어가거나, 2초를 기다림
    if verbose: print('driver load complete')

    # scroller
    scroll_down(driver, n_scroll=n_scroll, sleep_time=2, verbose=verbose)

    # 뉴스 목록 생성
    title_list = []
    url_list = []
    press_list = []

    press_elements = driver.find_elements(
        'css selector', 
        'a.info.press')
    title_elements = driver.find_elements(
        'css selector', 
        'a.news_tit')

    for title_el, press_el in zip(title_elements,
                                press_elements):
        title = title_el.text
        url = title_el.get_attribute('href')
        press = press_el.text.replace('언론사 선정', '')
        title_list.append(title)
        url_list.append(url)
        press_list.append(press)
    if verbose: print(f'{len(title_list)} news urls listed up..')

    # driver 종료
    driver.close()

    # 뉴스 데이터프레임 생성
    df = pd.DataFrame({'url': url_list,
                    'press': press_list,
                    'title': title_list,
                    })
    if verbose: print(f'Data Length: {len(df)}')

    return df


def get_news_contents(url_list, verbose=False):

    # 뉴스 본문 및 일시 추출
    text_list = []
    datetime_list = []
    urls_done = dict()

    for url in url_list:
        if verbose: print(url, end=' --- ')
        if url not in urls_done.keys():
            try:
                _, text, dt = get_news_title_text_date(url)
                time.sleep(0.5)
                if verbose: print('success')
            except ArticleException:
                text, dt = None, None
                if verbose: print('fail')
            urls_done[url] = (text, dt)
        else:
            if verbose: print('already done')
            text, dt = urls_done[url]
        text_list.append(text)
        datetime_list.append(dt)
    if verbose: print(f'{len(text_list)} news contents collected succesfully..')

    return text_list, datetime_list


if __name__ == '__main__':

    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    # ds = '2025.01.01'
    # de = '2025.03.31'
    n_scroll = 11
    
    if sys.argv[1] == '윤':
        # input
        keyword = '윤석열'
        outfname = './data/yoon_{}.csv'
    if sys.argv[1] == '이':
        # input
        keyword = '이재명'
        outfname = './data/lee_{}.csv'

    for m in range(1, 4):
        for d in range(1, 32):
            ds = de = f'2025.{m:02d}.{d:02d}'
            # 날짜 유효성 검사
            try:
                validate(ds.replace('.', '-'))
                print(f'Valid date: {ds}')
            except ValueError:
                print(f'Invalid date: {ds}')
                continue
            
            df = get_news_list(keyword, ds, de, n_scroll=n_scroll, 
                               headless=True, verbose=True)

            text_list, datetime_list = \
                get_news_contents(df['url'].tolist(), verbose=True)

            df['datetime'] = datetime_list
            df['text'] = text_list
            print(f'Data Length: {len(df)}')

            # 저장
            df.to_csv(outfname.format(ds), 
                    quoting=1, 
                    encoding='utf-8',
                    index=False)
        
            print('\n\n\n\n')