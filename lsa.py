# %% Load required libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap
from nltk.corpus import stopwords
from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt

# %%

class LSA():
    def __init__(self, data, tweets, n_clusters):
        self.data = data
        self.tweets = tweets
        self.num_clusters = n_clusters
        stop_words = set(stopwords.words('portuguese'))
        stop_words = stop_words.union(set(stopwords.words('english')))
        self.stop_words = stop_words.union(STOP_WORDS)
    
    def lsa(self):
        content_list = []
        for i in self.data:
            string_content = ""
            if "contents" in i:
                for all in i["contents"]:
                    if "content" in all:
                        string_content = string_content + str(all["content"])
                content_list.append(string_content)


        # tfidf vectorizer of scikit learn
        vectorizer = TfidfVectorizer(stop_words=self.stop_words,max_features=5000, max_df = 0.5, use_idf = True, ngram_range=(1,3))
        X = vectorizer.fit_transform([x[0] for x in self.tweets])
        print(X.shape) # check shape of the document-term matrix
        terms = vectorizer.get_feature_names()


        # num_clusters = 9
        km = KMeans(n_clusters=self.num_clusters)
        km.fit(X)
        self.clusters = km.labels_.tolist()


        U, Sigma, VT = randomized_svd(X, n_components=10, n_iter=100,
        random_state=122)


        for i, comp in enumerate(VT):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
            # print("Concept "+str(i)+": ")
            for t in sorted_terms:
                pass
                # print(t[0])
                # print(" ")


        X_topics=U*Sigma
        return umap.UMAP(n_neighbors=6, min_dist=0.5, random_state=12).fit_transform(X_topics)
            
    def plot(self):
        embedding = self.lsa()

        plt.figure(figsize=(7,5))
        plt.scatter(embedding[:, 0], embedding[:, 1], 
        c = self.clusters,
        s = 10, # size
        edgecolor='none'
        )
        plt.show()
