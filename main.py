# %%
import csv
import pandas as pd
from preprocess import Preprocess
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap
from nltk.corpus import stopwords
from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt


# %%
stop_words = set(stopwords.words('portuguese'))
stop_words = stop_words.union(set(stopwords.words('english')))
stop_words = stop_words.union(STOP_WORDS)

# %%
files = os.listdir('.\\data')
tweets = list()
df = pd.DataFrame()
seen_id = set()
files
# %%
for f in files:
    with open('.\\data\\'+f, 'r', newline='',encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweets.append((row['text'], row['id']))

    
    p = Preprocess()
    preprocessed_tweets = list()
    for t in tweets:
        if t[1] not in seen_id:
            x = p.preprocess_pipeline(t[0])
            preprocessed_tweets.append((x, t[1]))
            seen_id.add(t[1])


    preprocessed_tweets = [i for i in preprocessed_tweets if i[0] and i[0] != ' '] 
    # preprocessed_tweets[0]

    # Calling DataFrame constructor on list with columns specified 
    df = pd.DataFrame(preprocessed_tweets, columns =['Tweets', 'ID']) 
df.head(10)

# %%
df.to_csv('.\\data\\preprocessed_data.csv', sep=';')
df.to_excel(".\\data\\preprocessed_data.xlsx") 

# %%
df.head(10)

# %%
data = json.loads(json.dumps([x[0] for x in preprocessed_tweets]))

# %%
content_list = []
for i in data:
    string_content = ""
    if "contents" in i:
       for all in i["contents"]:
          if "content" in all:
             string_content = string_content + str(all["content"])
       content_list.append(string_content)

# %%
# tfidf vectorizer of scikit learn
vectorizer = TfidfVectorizer(stop_words=stop_words,max_features=10000, max_df = 0.5, use_idf = True, ngram_range=(1,3))
X = vectorizer.fit_transform([x[0] for x in preprocessed_tweets])
print(X.shape) # check shape of the document-term matrix
terms = vectorizer.get_feature_names()

# %%
num_clusters = 9
km = KMeans(n_clusters=num_clusters)
km.fit(X)
clusters = km.labels_.tolist()

# %%
U, Sigma, VT = randomized_svd(X, n_components=10, n_iter=100,
 random_state=122)

# %%
for i, comp in enumerate(VT):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    # print("Concept "+str(i)+": ")
    for t in sorted_terms:
        pass
        # print(t[0])
        # print(" ")

# %%
X_topics=U*Sigma
embedding = umap.UMAP(n_neighbors=6, min_dist=0.5, random_state=12).fit_transform(X_topics)

# %%
plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1], 
c = clusters,
s = 10, # size
edgecolor='none'
)
plt.show()

# %%
n = 0
for t in preprocessed_tweets:
    n += len(t[0].split()) 

print(n)
print(len(preprocessed_tweets))
print(n/len(preprocessed_tweets))
# %%
