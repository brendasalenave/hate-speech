# -*- coding: utf-8 -*-
# %%
# Load required libraries

# %%
# Load required libraries

import os
import csv
import re
import unicodedata
import string
from nltk.corpus import stopwords
from spacy.lang.pt.stop_words import STOP_WORDS
from nltk.tokenize import word_tokenize
import string

# %%
class Preprocess():
    def __init__(self, flag):
        self.flag = flag

    def stopwords_removal(self, text):
        stop_words = set(stopwords.words('portuguese'))
        stop_words = stop_words.union(set(stopwords.words('english')))
        stop_words = stop_words.union(STOP_WORDS)
        stop_words = stop_words.union(set(['ja','ate', 'vc','vcs','pra', 'ta','dia','ai','la','pq','cara','gente','voce','sao']))
        # print(type(stop_words))
        
        separator = ' '
        tokens = word_tokenize(text)
        filtered_sentence = [w for w in tokens if not w in stop_words]
        row = ' '.join(filtered_sentence)
        return row
    
    # Remove accents from tweets
    def accents_removal(self, text):
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

    # Remove hashtags from tweets
    def hashtag_removal(self, text):
        p = re.compile(r'#[\w|_]+', re.ASCII)
        return p.sub('',text) 
    
    # Remove mentions from tweets
    def mentions_removal(self, text):
        p = re.compile(r'@[\w|_]+', re.ASCII)
        return p.sub('',text) 
        
    # Normalize case    
    def to_lower(self, text):
        return text.lower()

    # Remove punctuations
    def punctuation_removal(self, text):
        # return ' '.join(word.strip(string.punctuation) for word in text.split())
        p = re.compile(r'[^\w\s]')
        return p.sub(' ',text) 
    
    def remove_laughs(self,text):
        text = ' ' + text + ' '
        text = re.sub(r'(\s)(kkkkkkkkk)(\s)',' ', text)
        p = re.compile(r'(\s)(kk+)(\s)')
        text = p.sub(' ',text)
        p = re.compile(r'(\s)(kakkakakakkak)\s|\s(k+a+)+(\s)')
        text = p.sub(' ',text)
        p = re.compile(r'(\s)([hah?]{2,}|[heh?]{2,}|[hih?]{2,}|[hoh?]{2,}|[huh?]{2,})(\s)')
        text = p.sub(' ',text)
        p = re.compile(r'(\s)([hua]{2,}|[hue]{2,})(\s)')
        text = p.sub(' ',text)
        # p = re.compile(r'kk+')
        # text = p.sub(' ',text)
        return text

    def remove_lengh_one_word(self, text):
        return ' '.join([w for w in text.split() if len(w)>1])
        
    def replace_multiple_spaces(self, text):
        return ' '.join(text.split())

    def links_removal(self, text):
        text = re.sub(r'pic\.twitter\.com/\w+\'','',text)
        #p = re.compile(r'^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&\'\(\)\*\+,;=.]+$')
        p = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        return p.sub('',text)

    def digits_removal(self, text):
        return re.sub(r'(\d)+','',text)

    def preprocess_pipeline(self, text):
        text = self.to_lower(text)
        text = self.links_removal(text)
        text = self.accents_removal(text)
        text = self.hashtag_removal(text)
        text = self.mentions_removal(text)
        if self.flag:
            text = self.stopwords_removal(text)
        text = self.punctuation_removal(text)
        text = self.digits_removal(text)
        text = self.remove_laughs(text)
        text = self.remove_lengh_one_word(text)
        text = self.replace_multiple_spaces(text)
        return text


# %%
