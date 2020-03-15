# %% Load required libraries
import gensim
import pyLDAvis.gensim
from gensim import corpora, models

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

from pprint import pprint

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from wordcloud import WordCloud
from collections import Counter

from sklearn.manifold import TSNE
import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import export_png
from bokeh.io import output_notebook
import numpy as np

import pandas as pd

# %%

class LDA():
    def __init__(self, tweets, clusters):
        self.tweets = tweets
        self.num_topics = clusters
    
    def process(self):
        data_words = [x[0].split() for x in self.tweets]

        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
        self.bigram_mod = gensim.models.phrases.Phraser(bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(trigram)

        self.data_ready = self.process_words(data_words)  # processed Text Data!

    def process_words(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
        texts = [[word for word in simple_preprocess(str(doc))] for doc in texts]
        texts = [self.bigram_mod[doc] for doc in texts]
        texts = [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]
        texts_out = []
        nlp = spacy.load("pt_core_news_sm", disable=['parser', 'ner'])
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            #texts_out.append([token.orth_ for token in doc if token.pos_ in allowed_postags])
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

        # remove stopwords once more after lemmatization
        texts_out = [[word for word in simple_preprocess(str(doc))] for doc in texts_out]    
        return texts_out


    def train_model(self):
        self.process()

        # Create Dictionary
        id2word = corpora.Dictionary(self.data_ready)

        # Create Corpus: Term Document Frequency
        self.corpus = [id2word.doc2bow(text) for text in self.data_ready]

        # Build LDA model
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                id2word=id2word,
                                                num_topics=4, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)

        pprint(self.lda_model.print_topics())

    def wordcloud_topics(self):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='white',
                        width=2500,
                        height=1800,
                        max_words=30,
                        colormap='tab10',
                        color_func=lambda *args, **kwargs: cols[i],
                        prefer_horizontal=1.0)

        topics = self.lda_model.show_topics(formatted=False)

        fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
            plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.savefig('wordcloud-topics', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
        plt.show()

    def wordcount_topics(self):
        topics = self.lda_model.show_topics(formatted=False)
        data_flat = [w for w_list in self.data_ready for w in w_list]
        counter = Counter(data_flat)

        out = []
        for i, topic in topics:
            for word, weight in topic:
                out.append([word, i , weight, counter[word]])

        df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

        # Plot Word Count and Weights of Topic Keywords
        fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        for i, ax in enumerate(axes.flatten()):
            ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
            ax_twin = ax.twinx()
            ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
            ax.set_ylabel('Word Count', color=cols[i])
            #ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 20000000)
            ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 30000)
            ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
            ax.tick_params(axis='y', left=False)
            ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
            ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

        fig.tight_layout(w_pad=2)    
        fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    

        plt.savefig('wordcount-topics', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
        plt.show()

    def sentences_chart(self, start = 0, end = 13):
        lda_model = self.lda_model
        corpus = self.corpus
        corp = corpus[start:end]
        mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

        fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)       
        axes[0].axis('off')
        for i, ax in enumerate(axes):
            if i > 0:
                corp_cur = corp[i-1] 
                topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
                word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]    
                ax.text(0.01, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',
                        fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

                # Draw Rectange
                topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
                ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1, 
                                    color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

                word_pos = 0.06
                for j, (word, topics) in enumerate(word_dominanttopic):
                    if j < 14:
                        ax.text(word_pos, 0.5, word,
                                horizontalalignment='left',
                                verticalalignment='center',
                                fontsize=16, color=mycolors[topics],
                                transform=ax.transAxes, fontweight=700)
                        word_pos += .009 * len(word)  # to move the word for the next iter
                        ax.axis('off')
                ax.text(word_pos, 0.5, '. . .',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, color='black',
                        transform=ax.transAxes)       

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=22, y=0.95, fontweight=700)
        plt.tight_layout()
        plt.savefig('sentences-colored', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        plt.show()

    def topics_per_document(self, model, corpus, start=0, end=1):
        corpus_sel = corpus[start:end]
        dominant_topics = []
        topic_percentages = []
        for i, corp in enumerate(corpus_sel):
            topic_percs, wordid_topics, wordid_phivalues = model[corp]
            dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
            dominant_topics.append((i, dominant_topic))
            topic_percentages.append(topic_percs)
        return(dominant_topics, topic_percentages)


    def documents_topics_weightage(self):
        # Sentence Coloring of N Sentences

        dominant_topics, topic_percentages = self.topics_per_document(model=self.lda_model, corpus=self.corpus, end=-1)            

        # Distribution of Dominant Topics in Each Document
        df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
        dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
        df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

        # Total Topic Distribution by actual weight
        topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
        df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

        # Top 3 Keywords for each Topic
        topic_top3words = [(i, topic) for i, topics in self.lda_model.show_topics(formatted=False) 
                                        for j, (topic, wt) in enumerate(topics) if j < 3]

        df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
        df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
        df_top3words.reset_index(level=0,inplace=True)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)

        # Topic Distribution by Dominant Topics
        ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
        ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
        tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
        ax1.xaxis.set_major_formatter(tick_formatter)
        ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
        ax1.set_ylabel('Number of Documents')
        ax1.set_ylim(0, 1500)

        # Topic Distribution by Topic Weights
        ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
        ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
        ax2.xaxis.set_major_formatter(tick_formatter)
        ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))

        plt.savefig('documents-topic-weightage', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        plt.show()

    def t_sne_clustering(self):
        topic_weights = []
        for i, row_list in enumerate(self.lda_model[self.corpus]):
            topic_weights.append([w for i, w in row_list[0]])

        # Array of topic weights    
        arr = pd.DataFrame(topic_weights).fillna(0).values

        # Keep the well separated points (optional)
        arr = arr[np.amax(arr, axis=1) > 0.35]

        # Dominant topic number in each doc
        topic_num = np.argmax(arr, axis=1)

        # tSNE Dimension Reduction
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(arr)

        # Plot the Topic Clusters using Bokeh
        output_notebook()
        n_topics = 4
        mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
        plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
                    plot_width=900, plot_height=700)
        plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
        
        # export_png(plot, filename="bokeh_plot.png")
        show(plot)

    def lda_plots(self):
        self.train_model()
        self.wordcloud_topics()
        self.wordcount_topics()
        self.sentences_chart()
        self.documents_topics_weightage()
        self.t_sne_clustering()
        self.lda_vis()

    def lda(self):
        l = [x[0].split() for x in self.tweets]
        dictionary = corpora.Dictionary(l)
        corpus = [dictionary.doc2bow(text) for text in self.tweets]

        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=self.num_topics, id2word = dictionary, passes=500, random_state=1)
        #print(ldamodel)
        #print(ldamodel.print_topics(num_topics=self.num_topics, num_words=4))
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
        pyLDAvis.display(lda_display)

        self.lda_save(ldamodel, corpus, dictionary)
    
    def lda_vis(self):
        
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(self.lda_model, self.corpus, dictionary=self.lda_model.id2word)
        pyLDAvis.display(vis) #vis

        self.lda_save()

    def lda_save(self):
        lda_display = pyLDAvis.gensim.prepare(self.lda_model, self.corpus, self.lda_model.id2word, sort_topics=False)
        pyLDAvis.display(lda_display)
        pyLDAvis.save_html(lda_display, 'lda.html')