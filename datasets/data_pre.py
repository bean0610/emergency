import re
import os
import pandas as pd
import numpy as np
# from hanspell import spell_checker
import csv

def preprocess(filename):
    path = "/home/nlplab/hdd1/AI_2022/KangSeungHyun/KDTL/datasets"
    f_list = os.listdir(path)
    print(f_list)

    if filename in f_list:
        df = pd.read_csv(os.path.join(path, filename), sep='\t', index_col=0)
        print(df)

    df['src'] = df['src'].str.replace(r'[()]',"", regex=True)
    df['tgt'] = df['tgt'].str.replace(r'[()]',"", regex=True)
    df['src'] = df['src'].str.replace('&quot;', "")
    df['tgt'] = df['tgt'].str.replace('&quot;', "")
    df['src'] = df['src'].str.replace('&apos;',"")
    df['tgt'] = df['tgt'].str.replace('&apos;',"")    
    print(df)
    
    # 중복 행 삭제
    # https://wikidocs.net/154060
    drop_dup = df.drop_duplicates()

    print(drop_dup)
    # # 불필요한 내용 삭제
    # # 공백제거
    # blank_s= strip_a.strip()
    # print(blank_s)
    return df



data = preprocess("train_esko.tsv")

print(data)

data.to_csv("pre_train_esko.tsv", sep = '\t', index=False)


def preprocess2(filename):
    path = "/home/nlplab/hdd1/AI_2022/KangSeungHyun/Triangular_translation/language_files/mono"
    f_list = os.listdir(path)
    # print(f_list)

    if filename in f_list:
        df = pd.read_csv(os.path.join(path, filename), sep='\t', header = None)
        df.columns = ["ID", "lang", "txt"]
        # print(df)
        df = df[df['txt'].str.len() >= 5]
        txt = df.loc[ :, "txt"]
        # print(txt)
        # print(len(txt))
        max_length = max(txt, key=lambda x: len(x))
        # print(len(max_length))
        min_length = min(txt, key=lambda x: len(x))
        # print(len(min_length))
        txt.to_csv("pre_" + filename, sep="\t")
        

# preprocess2("eng.tsv")