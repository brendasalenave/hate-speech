# %% Load required libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import cufflinks as cf

#%%
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

class VIS():
    def __init__(self, df):
        self.df = df
    
    def get_top_n_gram(self, corpus, n_gram,n=None):
        vec = CountVectorizer(ngram_range=(n_gram, n_gram)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]
    
    def vis_top_n_grams(self, n_gram):
        common_words = self.get_top_n_gram(self.df['Tweets'], n_gram,20)
        # for word, freq in common_words:
        #     print(word, freq)
        df4 = pd.DataFrame(common_words, columns = ['Tweets' , 'count'])
        # df4.groupby('Tweets').sum()['count'].sort_values(ascending=False).iplot(
        #     kind='bar', yTitle='Count', linecolor='black', layout={'xaxis_title':"Words", 'yaxis_title':"Count", 'margin':{'l': 0, 'r': 0, 't': 0, 'b': 0},'xaxis': {'automargin': True}, 'yaxis': {'automargin': True}})
        df4.groupby('Tweets').sum()['count'].sort_values(ascending=False).iplot(
           kind='bar', yTitle='Count', linecolor='black', layout={'paper_bgcolor':'rgba(0,0,0,0)', 'plot_bgcolor':'rgba(0,0,0,0)', 'xaxis': {'automargin': True, 'gridcolor':'lightgray'}, 'yaxis': {'automargin': True, 'title':'Count', 'gridcolor':'lightgray'}})


    def get_top_n_words(self, corpus, n=None):
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]
    
    def vis_top_words(self):
        common_words = self.get_top_n_words(self.df['Tweets'], 20)
        # for word, freq in common_words:
        #     print(word, '\t',freq)
        df_12 = pd.DataFrame(common_words, columns = ['Tweets' , 'count'])
        df_12.groupby('Tweets').sum()['count'].sort_values(ascending=False).iplot(
           kind='bar', yTitle='Count', linecolor='black', layout={'paper_bgcolor':'rgba(0,0,0,0)', 'plot_bgcolor':'rgba(0,0,0,0)', 'xaxis': {'automargin': True, 'gridcolor':'lightgray'}, 'yaxis': {'automargin': True, 'title':'Count', 'gridcolor':'lightgray'}})
        
        #kind='bar', yTitle='Count', xTitle='Words',linecolor='black', margin=(0,0,400,0),layout={'margin':{'l': 0, 'r': 0, 't': 0, 'b': 0},'xaxis': {'automargin': True}, 'yaxis': {'automargin': True}})

