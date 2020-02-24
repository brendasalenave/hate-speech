# -*- coding: utf-8 -*-
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
    def __init__(self):
        pass

    def stopwords_removal(self, text):
        stop_words = set(stopwords.words('portuguese'))
        stop_words = stop_words.union(set(stopwords.words('english')))
        stop_words = stop_words.union(STOP_WORDS)
        # print(type(stop_words))
        
        separator = ' '
        tokens = word_tokenize(text)
        filtered_sentence = [w for w in tokens if not w in stop_words]
        row = ' '.join(filtered_sentence)
        return row
    
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
        return p.sub('',text) 
        
    def replace_multiple_spaces(self, text):
        return ' '.join(text.split())

    def preprocess_pipeline(self, text):
        text = self.to_lower(text)
        text = self.hashtag_removal(text)
        text = self.mentions_removal(text)
        text = self.stopwords_removal(text)
        text = self.punctuation_removal(text)
        text = self.replace_multiple_spaces(text)
        return text


# %%
