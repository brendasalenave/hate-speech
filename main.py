# %%
import csv
import pandas as pd
from preprocess import Preprocess
from lsa import LSA
from lda import LDA
import os
import json
from visualization import VIS
from sklearn.feature_extraction.text import CountVectorizer
import cufflinks as cf
from textblob import TextBlob
import spacy

# %%
if os.path.exists("./data/preprocessed_data.csv"):
  os.remove("./data/preprocessed_data.csv")
  os.remove("./data/preprocessed_data.xlsx")

# %%
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

# %%
files = os.listdir('./data')
tweets = list()
df_1 = pd.DataFrame() # preprocessados
df_2 = pd.DataFrame() # originais
seen_id = set()
files
# %%
p1 = Preprocess(1) # remove stopwords
p2 = Preprocess(0) # nao remove stopwords
preprocessed_tweets = list()
original_tweets = list()
# %%
for f in files:
    with open('./data/'+f, 'r', newline='',encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweets.append((row['text'], row['id']))
    
    p_tweets = list()
    o_tweets = list()
    for t in tweets:
        if t[1] not in seen_id:
            x = p1.preprocess_pipeline(t[0])
            p_tweets.append((x, t[1]))
            seen_id.add(t[1])
            x = p2.preprocess_pipeline(t[0])
            o_tweets.append((x, t[1]))


    preprocessed_tweets = preprocessed_tweets + [i for i in p_tweets if i[0] and i[0] != ' '] 
    original_tweets = original_tweets + [i for i in o_tweets if i[0] and i[0] != ' '] 
    
    # preprocessed_tweets[0]

    # Calling DataFrame constructor on list with columns specified 
    df_1 = pd.DataFrame(preprocessed_tweets, columns =['Tweets', 'ID']) 
    df_2 = pd.DataFrame(original_tweets, columns =['Tweets', 'ID'])
df_2.head(10)

# %%
df_1.to_csv('./data/preprocessed_data.csv', sep=';')
df_1.to_excel("./data/preprocessed_data.xlsx") 

# %%
df_1.head(10)
df_1.size
# %%
data = json.loads(json.dumps([x[0] for x in preprocessed_tweets]))

#%%

#lsa = LSA(data, tweets, 5)
#lsa.plot()

# %%
lda = LDA(preprocessed_tweets, 5)
lda.lda()

# %%
# Without stopwords
v = VIS(df_1)
v.vis_top_words()
print('2grams')
v.vis_top_n_grams(2)
print('3grams')
v.vis_top_n_grams(3)

# %%
print('POS Tag')
v.vis_pos()

# %%
# With stopwords
v = VIS(df_2)
v.vis_top_words()
v.vis_top_bigrams()

# %%

# %%
