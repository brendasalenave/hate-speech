# %% Load required libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import cufflinks as cf

#%%
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

class VIS():
    def __init__(self, df):
        self.df = df
        self.nlp = spacy.load('pt_core_news_sm')

    
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
           kind='bar', yTitle='Count', linecolor='black', layout={'paper_bgcolor':'rgba(0,0,0,0)', 'plot_bgcolor':'rgba(0,0,0,0)', 'xaxis': {'automargin': True, 'gridcolor':'ghostwhite'}, 'yaxis': {'automargin': True, 'title':'Count', 'gridcolor':'ghostwhite'}})


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
           kind='bar', yTitle='Count', linecolor='black', layout={'paper_bgcolor':'rgba(0,0,0,0)', 'plot_bgcolor':'rgba(0,0,0,0)', 'xaxis': {'automargin': True, 'gridcolor':'ghostwhite'}, 'yaxis': {'automargin': True, 'title':'Count', 'gridcolor':'ghostwhite'}})
        
        #kind='bar', yTitle='Count', xTitle='Words',linecolor='black', margin=(0,0,400,0),layout={'margin':{'l': 0, 'r': 0, 't': 0, 'b': 0},'xaxis': {'automargin': True}, 'yaxis': {'automargin': True}})

    def vis_pos(self):
        tweets = list(self.df['Tweets'])
        l_tweets = [l.split() for l in tweets]
        flat_list = [item for sublist in l_tweets for item in sublist]
        l_tweets = [[(token.orth_, token.lemma_,token.pos_) for token in self.nlp(s)] for s in flat_list]
        flat_list = [item for sublist in l_tweets for item in sublist]
        df_pos = pd.DataFrame.from_records(flat_list, columns =['Word', 'Lemma', 'POS']) 

        pos_df = df_pos.POS.value_counts()[:20]
        pos_df.iplot(
            kind='bar',
            xTitle='POS',
            yTitle='Count',
            layout={'paper_bgcolor':'rgba(0,0,0,0)', 'plot_bgcolor':'rgba(0,0,0,0)', 'xaxis': {'automargin': True, 'gridcolor':'ghostwhite'}, 'yaxis': {'automargin': True, 'title':'Count', 'gridcolor':'ghostwhite'}})
            #title='Top 20 Part-of-speech tagging for review corpus')

