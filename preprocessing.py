#!/usr/bin/envpython3
# -*- coding: utf-8 -*-

import os
import csv
import re
import unicodedata
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocess(text,stop_words):
    raw = text.lower()
    # raw = unicodedata.normalize('NFKD', raw).encode('ASCII', 'ignore')
    # raw = (str(raw).replace('b\"','').replace('\'',''))
    #raw = re.sub(r'(＃|#) ','#',raw,re.UNICODE)
    #raw = re.sub(r'(＠|@) ','@',raw,re.UNICODE)
    raw = re.sub(r'pic\.twitter\.com/\w+\'','',raw)
    raw = re.sub(r'^b|\'|\"','',raw)

    separator = ' '
    word_tokens = word_tokenize(raw)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    raw = separator.join(filtered_sentence)

    raw = raw.replace('# ', "#").replace('@ ', "@").replace('\( ', '(').replace('\) ', ')')
    raw = re.sub(r'\.|\,|\!|\?','',raw)

    return raw

def unifica():
    for f in os.listdir('./data-preprocessed'):
        with open('./data-preprocessed/'+f, 'r', newline='',encoding='utf-8') as csvfile:
            #print('./data-preprocessed/'+f)
            with open('set_final.csv', 'a', newline='',encoding='utf-8') as outputfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                writer = csv.writer(outputfile, delimiter=',', quotechar='"')

                for row in reader:
                    writer.writerow(row)


def main():
    stop_words = set(stopwords.words('portuguese'))

    for f in os.listdir('./data-utf8'):
        # print(f)
        with open('./data-utf8/'+f,'r',newline='',encoding='utf-8') as csvfile:
            new_file = f.replace('-utf8','preprocessed')
            seen = set()
            with open('./data-preprocessed/'+new_file, 'w', newline='',encoding='utf-8') as outputfile:
                reader = csv.reader(csvfile, delimiter=';', quotechar='"')
                writer = csv.writer(outputfile, delimiter=',', quotechar='"')
                for row in reader:
                    if row[4] == '':
                        row[4] = row[3]
                        row[3] = '-'

                    row[4] = preprocess(row[4],stop_words)
                    if row[4] in seen: continue # skip duplicate
                    seen.add(row[4])
                    writer.writerow(row)
    #break
    unifica()


if __name__ == '__main__':
    main()
