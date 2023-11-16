import pandas as pd
import os
import json

def read_file(filename):
    path = "/home/nlplab/hdd1/AI_2022/KangSeungHyun/KDTL/datasets/processed/KD"
    f_list = os.listdir(path)
    print(f_list)

    if filename in f_list:
        df = pd.read_csv(os.path.join(path, filename), sep='\t', index_col=0)
        print(df)
        return df

data = read_file("pre_train_ko_es.tsv")
korean = data["src"]
# print(korean)
spanish = data["tgt"]
# data1 = read_file("pre_test_ko_es.tsv")
# korean1 = data["src"]
# # print(korean1)
# spanish1 = data["tgt"]
# data2 = read_file("pre_dev_ko_es.tsv")
# korean2 = data["src"]
# # print(korean2)
# spanish2 = data["tgt"]
# korean_merged = []
# spanish_merged = []

# korean_merged = korean.append(korean1)
# print(korean_merged)
# print(spanish)
# # with open(FILE_PATH, 'w') as output_file:


# list = []
# for es, kor in zip(spanish, korean):
#     list.append({"translation": {"es": es, "kor": kor}})
# print(list)


# with open("es_ko_train.json", 'a',encoding='utf-8') as file:
#     for translation in list:
#         json.dump(translation, file, ensure_ascii=False)
#         file.write('\n')

