import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LdaSeqModel
import gensim
import logging
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

df = pd.read_csv("./data/CNN_Articels_clean/CNN_Articels_clean.csv")
df.head()


# preprocessing1 ===================================================

target_months = ['2021-11', '2021-12', '2022-01', '2022-02', '2022-03']
df['Date published'] = pd.to_datetime(df['Date published'])
df['Date published'] = df['Date published'].dt.to_period('M').astype(str)

data = df[['Date published', 'Article text']]
data.head()
data = data[data['Date published'].isin(target_months)]
data.head()

corpus = data["Article text"]
spacy.cli.download('en_core_web_sm')
nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])

# removing stopword, punc and digit tokens. alphanumeric word such as Covid-19 are retained
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop
              and not token.is_punct and not token.is_digit]
    return tokens


corpus = corpus.map(preprocess_text)
print(corpus.iloc[0])

# gensim dictionary ---------------------------------------
dictionary = corpora.Dictionary(corpus)
dictionary.filter_extremes(no_below=40, no_above=0.9)
print(dictionary.token2id)

# creating BoW ------------------------------------------------------
# bow_representation = CountVectorizer(max_df=0.80, tokenizer=preprocess_text ,min_df=0.01)
# bow_transform = bow_representation.fit_transform(corpus["Article text"])

bow_rep = corpus.apply(lambda x: dictionary.doc2bow(x))
print(bow_rep)

# time slice --------------------------------------------------------------
article_count_per_month = data.groupby('Date published').size()
print(article_count_per_month)
time_slice = article_count_per_month.tolist()
print(time_slice)

# SeqLDA modeling ========================================================

ldaseq = gensim.models.ldaseqmodel.LdaSeqModel(corpus=bow_rep, id2word=dictionary,
                                         time_slice=time_slice, num_topics=7,
                                         random_state=123, chunksize=100)
# print("done")

# ldaseq.save('./research-note/ldaseq_model')

ldaseq.print_topics(time=0)
ldaseq.print_topics(time=1)
ldaseq.print_topics(time=2)
ldaseq.print_topics(time=3)
ldaseq.print_topics(time=4)

# ldaseq = gensim.models.ldaseqmodel.LdaSeqModel.load('.//research-note/ldaseq_model')

# LDASeq model evaluation ===================================

# UMass score ----------------------------------------------
timeslice1_topics = ldaseq.dtm_coherence(time=0)

coherence_scores = []

for time in range(len(ldaseq.time_slice)):
    timeslice_topics = ldaseq.dtm_coherence(time=time)
    coherence_model = CoherenceModel(topics=timeslice_topics, corpus=bow_rep,
                                     dictionary=dictionary, coherence='u_mass')
    coherence_score = coherence_model.get_coherence()
    coherence_scores.append(coherence_score)

average_coherence = sum(coherence_scores) / len(coherence_scores)
print(average_coherence)

# coherence_cv_score ----------------------------------
coherence_scores_cv = []

for time in range(len(ldaseq.time_slice)):
    timeslice_topics_cv = ldaseq.dtm_coherence(time=time)
    coherence_model_cv = CoherenceModel(topics=timeslice_topics_cv, texts=corpus,
                                        corpus=bow_rep, dictionary=dictionary, coherence='c_v')
    coherence_score_cv = coherence_model_cv.get_coherence()
    coherence_scores_cv.append(coherence_score_cv)

average_coherence_cv = sum(coherence_scores_cv) / len(coherence_scores_cv)
print(average_coherence_cv)

# preprocessing2 ======================================

start_date = '2021-11-01'
end_date = '2022-03-31'
filtered_df = df[(df['Date published'] >= start_date) & (df['Date published'] <= end_date)]
timestamps = filtered_df['Date published'].tolist()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

# removing stopwords and empty strings and transforming words to lower case
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


filtered_df['Article text'] = filtered_df['Article text'].apply(remove_stopwords)
text = filtered_df['Article text'].tolist()
text = [word for word in text if word != ""]

timestamps = df['Date published'].tolist()
print(timestamps)

# BERTopic modeling =================================================

topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(text)
topics_over_time = topic_model.topics_over_time(text, timestamps, datetime_format="%b%M", nr_bins=20)


# Visualization ===============================================

doc_topic, topic_term, doc_lengths, term_frequency, vocab = ldaseq.dtm_vis(time=0, corpus=bow_rep)
vis_wrapper = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths, vocab=vocab, term_frequency=term_frequency)
pyLDAvis.save_html(vis_wrapper, 'lda_visualization.html')


fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=7)
fig.write_html("bertopic_vis.html")

# fig2 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
# fig2.write_html("bertopic_vis20.html")

# fig3 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
# fig3.write_html("bertopic_vis_alldata.html")  # Save to an HTML file

