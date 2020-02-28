# %% Load required libraries
import gensim
import pyLDAvis.gensim
from gensim import corpora, models

# %%

class LDA():
    def __init__(self, tweets, clusters):
        self.tweets = tweets
        self.num_topics = clusters
    
    def lda(self):
        l = [x[0].split() for x in self.tweets]
        dictionary = corpora.Dictionary(l)
        corpus = [dictionary.doc2bow(text) for text in self.tweets]

        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=self.num_topics, id2word = dictionary, passes=100, random_state=1)
        #print(ldamodel)
        #print(ldamodel.print_topics(num_topics=self.num_topics, num_words=4))
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
        pyLDAvis.display(lda_display)

        self.lda_save(ldamodel, corpus, dictionary)

    def lda_save(self, ldamodel, corpus, dictionary):
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
        pyLDAvis.display(lda_display)
        pyLDAvis.save_html(lda_display, 'lda.html')