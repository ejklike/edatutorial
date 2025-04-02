import os
import sys
import datetime

import pandas as pd


def validate(date_text):
    try:
        datetime.date.fromisoformat(date_text)
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")


if __name__ == '__main__':

    if sys.argv[1] == '윤':
        # input
        keyword = '윤석열'
        outfname = './data/yoon_{}.csv'
    if sys.argv[1] == '이':
        # input
        keyword = '이재명'
        outfname = './data/lee_{}.csv'

    df_list = []

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

            if os.path.exists(outfname.format(ds)):
                df = pd.read_csv(outfname.format(ds), encoding='utf-8')
                df_list.append(df)
                print(f'Data Length: {len(df)}')
    
    pd.concat(df_list, ignore_index=True).to_csv(
            outfname.format('all'),
            quoting=1, 
            encoding='utf-8',
            index=False)