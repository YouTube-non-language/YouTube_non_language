import nltk
from collections import Counter
import plotly.graph_objects as go
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

from data_s3 import DataProcessor


class FigureGenerator:
    def __init__(self, top_route, top_name, 
                       other1_route, other1_name,
                       other2_route, other2_name, 
                       user_route, user_name):
        self.data_processor = DataProcessor()
        print('~~ Data Load Start ~~')
        self.top = self.data_processor.top_data(top_route, top_name)
        self.other1 = self.data_processor.other1_data(other1_route, other1_name)
        self.other2 = self.data_processor.other2_data(other2_route, other2_name)
        self.user = self.data_processor.user_data(user_route, user_name)

    def process_text(self, text):
        return self.data_processor.text_preprocess(text)

    def create_word_freq_df(self, text):
        word_freq = Counter(self.process_text(text))
        df = pd.DataFrame.from_dict(word_freq, orient='index').reset_index()
        df = df.rename(columns={'index':'word', 0:'count'})
        df = df.sort_values(by='count', ascending=False).head(10)
        return df

    def word_freq(self):  
        '''
        Word Frequency
        '''
        df_top = self.create_word_freq_df(self.top)
        df_other1 = self.create_word_freq_df(self.other1)
        df_other2 = self.create_word_freq_df(self.other2)
        df_user = self.create_word_freq_df(self.user)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_top['word'], y=df_top['count'], name='Top Rank Lecture'))
        fig.add_trace(go.Bar(x=df_other1['word'], y=df_other1['count'], name='Other1'))
        fig.add_trace(go.Bar(x=df_other2['word'], y=df_other2['count'], name='Other2'))
        fig.add_trace(go.Bar(x=df_user['word'], y=df_user['count'], name='Your Video'))
        fig.update_layout(barmode='group', title_text="<b>Top 10 Most Used Words<b>")
        return fig
    
    def sentence_senti(self):
        '''
        Sentiment Analysis
        '''
        sia = SentimentIntensityAnalyzer()
        # top
        sentences_top = nltk.sent_tokenize(self.top)
        positive, negative, neutral = 0, 0, 0
        for sentence in sentences_top:
            sentiment = sia.polarity_scores(sentence)
            if sentiment['compound'] > 0.05:
                positive += 1
            elif sentiment['compound'] < -0.05:
                negative += 1
            else:
                neutral += 1
        df_top_sent = pd.DataFrame({'sentiment': ['positive', 'negative', 'neutral'], 
                                    'count': [positive, negative, neutral]})
        # other1
        sentences_other = nltk.sent_tokenize(self.other1)
        positive_other, negative_other, neutral_other  = 0, 0, 0         
        for sentence in sentences_other:
            sentiment = sia.polarity_scores(sentence)
            if sentiment['compound'] > 0.05:
                positive_other += 1
            elif sentiment['compound'] < -0.05:
                negative_other += 1
            else:
                neutral_other += 1
        df_other1_sent = pd.DataFrame({'sentiment': ['positive', 'negative', 'neutral'], 
                                      'count': [positive_other, negative_other, neutral_other]})
        # other2
        sentences_other = nltk.sent_tokenize(self.other2)
        positive_other, negative_other, neutral_other  = 0, 0, 0
        for sentence in sentences_other:
            sentiment = sia.polarity_scores(sentence)
            if sentiment['compound'] > 0.05:
                positive_other += 1
            elif sentiment['compound'] < -0.05:
                negative_other += 1
            else:
                neutral_other += 1
        df_other2_sent = pd.DataFrame({'sentiment': ['positive', 'negative', 'neutral'], 
                                      'count': [positive_other, negative_other, neutral_other]})
        # user
        sentences_other = nltk.sent_tokenize(self.user)
        positive_other, negative_other, neutral_other  = 0, 0, 0
        for sentence in sentences_other:
            sentiment = sia.polarity_scores(sentence)
            if sentiment['compound'] > 0.05:
                positive_other += 1
            elif sentiment['compound'] < -0.05:
                negative_other += 1
            else:
                neutral_other += 1
        df_user_sent = pd.DataFrame({'sentiment': ['positive', 'negative', 'neutral'], 
                                     'count': [positive_other, negative_other, neutral_other]})

        fig = go.Figure()
        fig = make_subplots(rows=1, cols=4, 
                            specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],
                            subplot_titles=['<b>Top<b>', '<b>Other1<b>', '<b>Other2<b>', '<b>Your Video<b>'])
        fig.add_trace(go.Pie(labels=df_top_sent['sentiment'], 
                     values=df_top_sent['count'], 
                     hole=.3,), row=1, col=1)
        fig.add_trace(go.Pie(labels=df_other1_sent['sentiment'], 
                     values=df_other1_sent['count'], 
                     hole=.3,), row=1, col=2)
        fig.add_trace(go.Pie(labels=df_other2_sent['sentiment'], 
                     values=df_other2_sent['count'], 
                     hole=.3), row=1, col=3)        
        fig.add_trace(go.Pie(labels=df_user_sent['sentiment'], 
                     values=df_user_sent['count'], 
                     hole=.3), row=1, col=4)

        return fig
    
    def n_grams(self):
        '''
        N-grams
        '''
        # Top Lecture
        bigrams = list(ngrams(self.process_text(self.top), 5))
        bigram_freq = Counter(bigrams)
        df_bigrams_top = pd.DataFrame.from_dict(bigram_freq, orient='index').reset_index()
        df_bigrams_top = df_bigrams_top.rename(columns={'index':'bigram', 0:'count'})
        df_bigrams_top = df_bigrams_top.sort_values(by='count', ascending=False)
        # Other Lecture1
        bigrams_other = list(ngrams(self.process_text(self.other1), 5))
        bigram_freq_other = Counter(bigrams_other)
        df_bigrams_other1 = pd.DataFrame.from_dict(bigram_freq_other, orient='index').reset_index()
        df_bigrams_other1 = df_bigrams_other1.rename(columns={'index':'bigram', 0:'count'})
        df_bigrams_other1 = df_bigrams_other1.sort_values(by='count', ascending=False)
        # Other Lecture2
        bigrams_other = list(ngrams(self.process_text(self.other2), 5))
        bigram_freq_other = Counter(bigrams_other)
        df_bigrams_other2 = pd.DataFrame.from_dict(bigram_freq_other, orient='index').reset_index()
        df_bigrams_other2 = df_bigrams_other2.rename(columns={'index':'bigram', 0:'count'})
        df_bigrams_other2 = df_bigrams_other2.sort_values(by='count', ascending=False)
        # User Video
        bigrams_user = list(ngrams(self.process_text(self.user), 5))
        bigram_freq_user = Counter(bigrams_user)
        df_bigrams_user = pd.DataFrame.from_dict(bigram_freq_user, orient='index').reset_index()
        df_bigrams_user = df_bigrams_user.rename(columns={'index':'bigram', 0:'count'})
        df_bigrams_user = df_bigrams_user.sort_values(by='count', ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_bigrams_top['bigram'], y=df_bigrams_top['count'], name='Top Rank Lecture'))
        fig.add_trace(go.Bar(x=df_bigrams_other1['bigram'], y=df_bigrams_other1['count'], name='Other Lecture'))
        fig.add_trace(go.Bar(x=df_bigrams_other2['bigram'], y=df_bigrams_other2['count'], name='Other Lecture'))
        fig.add_trace(go.Bar(x=df_bigrams_user['bigram'], y=df_bigrams_user['count'], name='Your Video'))
        fig.update_layout(barmode='group', title_text="<b>Bigram Frequency<b>")
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="")
        return fig


    def pos(self):
        '''
        Part of Speech (POS) Tagging
        '''
        pos_tags = nltk.pos_tag(self.process_text(self.top))
        df_tag_top = pd.DataFrame(pos_tags, columns=['word', 'POS'])
        # Create a mapping dictionary
        pos_mapping = {'NNP': 'Proper Noun,<br>Singular', 
                       'NN': 'Noun,<br>Singular or Mass', 
                       'VB': 'Verb,<br>Base Form', 
                       'VBZ': 'Verb,<br>3rd person<br>Singular Present', 
                       'JJ': 'Adjective', 
                       'DT': 'Determiner', 
                       'IN': 'Preposition'} 
        color_mapping = {'Proper Noun,<br>Singular': 'blue'}
        # Top Lecture
        df_tag_top['POS'] = df_tag_top['POS'].replace(pos_mapping)
        df_tag_top['count'] = df_tag_top.groupby('POS')['POS'].transform('count') # Count the frequency of each POS
        df_tag_top = df_tag_top.sort_values(by='count', ascending=False).drop_duplicates() # Sort by count
        df_tag_top1 = df_tag_top[df_tag_top['count'] > df_tag_top['count'].mean()] # Select the POS with count > mean
        df_tag_top2 = df_tag_top[df_tag_top['count'] <= df_tag_top['count'].mean()]  
        # Other Lecture
        pos_tags_other = nltk.pos_tag(self.process_text(self.other1))
        df_tag_other = pd.DataFrame(pos_tags_other, columns=['word', 'POS'])
        df_tag_other['POS'] = df_tag_other['POS'].replace(pos_mapping)
        df_tag_other['count'] = df_tag_other.groupby('POS')['POS'].transform('count')
        df_tag_other = df_tag_other.sort_values(by='count', ascending=False).drop_duplicates()
        df_tag_other1_1 = df_tag_other[df_tag_other['count'] > df_tag_other['count'].mean()]
        df_tag_other1_2 = df_tag_other[df_tag_other['count'] <= df_tag_other['count'].mean()]
        # Other2 Lecture
        pos_tags_other = nltk.pos_tag(self.process_text(self.other2))
        df_tag_other = pd.DataFrame(pos_tags_other, columns=['word', 'POS'])
        df_tag_other['POS'] = df_tag_other['POS'].replace(pos_mapping)
        df_tag_other['count'] = df_tag_other.groupby('POS')['POS'].transform('count')
        df_tag_other = df_tag_other.sort_values(by='count', ascending=False).drop_duplicates()
        df_tag_other2_1 = df_tag_other[df_tag_other['count'] > df_tag_other['count'].mean()]
        df_tag_other2_2 = df_tag_other[df_tag_other['count'] <= df_tag_other['count'].mean()]
        # User Video
        pos_tags_user = nltk.pos_tag(self.process_text(self.user))
        df_tag_user = pd.DataFrame(pos_tags_user, columns=['word', 'POS'])
        df_tag_user['POS'] = df_tag_user['POS'].replace(pos_mapping)
        df_tag_user['count'] = df_tag_user.groupby('POS')['POS'].transform('count')
        df_tag_user = df_tag_user.sort_values(by='count', ascending=False).drop_duplicates()
        df_tag_user1 = df_tag_user[df_tag_user['count'] > df_tag_user['count'].mean()]
        df_tag_user2 = df_tag_user[df_tag_user['count'] <= df_tag_user['count'].mean()]
        
        # Group by 'POS' and calculate the mean of 'count'
        df_tag_top1_grouped = df_tag_top1.groupby('POS', as_index=False)['count'].mean()
        df_tag_top2_grouped = df_tag_top2.groupby('POS', as_index=False)['count'].mean()
        df_tag_top2_grouped = df_tag_top2_grouped.sort_values('count', ascending=False)
        df_tag_other1_1_grouped = df_tag_other1_1.groupby('POS', as_index=False)['count'].mean()
        df_tag_other1_2_grouped = df_tag_other1_2.groupby('POS', as_index=False)['count'].mean()
        df_tag_other1_2_grouped = df_tag_other1_2_grouped.sort_values('count', ascending=False)
        df_tag_other1_2_grouped = df_tag_other1_2_grouped.head(5)
        df_tag_other2_1_grouped = df_tag_other2_1.groupby('POS', as_index=False)['count'].mean()
        df_tag_other2_2_grouped = df_tag_other2_2.groupby('POS', as_index=False)['count'].mean()
        df_tag_other2_2_grouped = df_tag_other2_2_grouped.sort_values('count', ascending=False)
        df_tag_other2_2_grouped = df_tag_other2_2_grouped.head(5)
        df_tag_user1_grouped = df_tag_user1.groupby('POS', as_index=False)['count'].mean()
        df_tag_user2_grouped = df_tag_user2.groupby('POS', as_index=False)['count'].mean()
        df_tag_user2_grouped = df_tag_user2_grouped.sort_values('count', ascending=False)
        df_tag_user2_grouped = df_tag_user2_grouped.head(5)        

        # Create subplot with two y-axes
        fig = make_subplots(subplot_titles=['<b>Top<b>', '<b>Other1<b>', '<b>Other2<b>','<b>Your Video<b>'], rows=2, cols=4)
        fig.add_trace(go.Bar(x=df_tag_top1_grouped['POS'], y=df_tag_top1_grouped['count'], text=df_tag_top1_grouped['count'], 
                            textposition='auto', marker_line_width=0), row=1, col=1) 
        fig.add_trace(go.Bar(x=df_tag_top2_grouped['POS'], y=df_tag_top2_grouped['count'], text=df_tag_top2_grouped['count'], 
                            textposition='auto', marker_line_width=0), row=2, col=1)
        fig.add_trace(go.Bar(x=df_tag_other1_1_grouped['POS'], y=df_tag_other1_1_grouped['count'], text=df_tag_other1_1_grouped['count'], 
                            textposition='auto', marker_line_width=0), row=1, col=2)
        fig.add_trace(go.Bar(x=df_tag_other1_2_grouped['POS'], y=df_tag_other1_2_grouped['count'], text=df_tag_other1_2_grouped['count'], 
                            textposition='auto', marker_line_width=0), row=2, col=2)
        fig.add_trace(go.Bar(x=df_tag_other2_1_grouped['POS'], y=df_tag_other2_1_grouped['count'], text=df_tag_other2_1_grouped['count'],
                            textposition='auto', marker_line_width=0), row=1, col=3)
        fig.add_trace(go.Bar(x=df_tag_other2_2_grouped['POS'], y=df_tag_other2_2_grouped['count'], text=df_tag_other2_2_grouped['count'],
                            textposition='auto', marker_line_width=0), row=2, col=3)
        fig.add_trace(go.Bar(x=df_tag_user1_grouped['POS'], y=df_tag_user1_grouped['count'], text=df_tag_user1_grouped['count'], 
                            textposition='auto', marker_line_width=0), row=1, col=4)
        fig.add_trace(go.Bar(x=df_tag_user2_grouped['POS'], y=df_tag_user2_grouped['count'], text=df_tag_user2_grouped['count'],
                            textposition='auto', marker_line_width=0), row=2, col=4)
        
        # fig.update_layout(title_text="<b>Part of Speech Tagging<b>")  # height=1200, width=800, 
        fig.update_layout(showlegend=False)
        return fig
    
    def similar(self):
        ncic = self.data_processor.ncic('dev/NCIC/', 'NCIC')
        # Convert the texts into TF-IDF features
        vectorizer = TfidfVectorizer().fit_transform([ncic, self.top, self.other1, self.other2, self.user])
        vectors = vectorizer.toarray()
        # Compute the cosine similarity
        cosine_sim = cosine_similarity(vectors)
        # Compute the Euclidean distance
        euclidean_distance_ncic_top = euclidean(vectors[0], vectors[1])
        euclidean_distance_ncic_other1 = euclidean(vectors[0], vectors[2])
        euclidean_distance_ncic_other2 = euclidean(vectors[0], vectors[3])
        euclidean_distance_ncic_user = euclidean(vectors[0], vectors[4])
        # Plot the results
        labels = ['NCIC-Top_Lecture', 'NCIC-Other1_Lecture', 'NCIC-Other2_Lecture', 'NCIC-Your_Video']
        cosine_values = [cosine_sim[0][1], cosine_sim[0][2], cosine_sim[0][3], cosine_sim[0][4]]
        euclidean_values = [euclidean_distance_ncic_top, euclidean_distance_ncic_other1, euclidean_distance_ncic_other2, euclidean_distance_ncic_user]
        fig = go.Figure(data=[
            go.Bar(name='Cosine Similarity', x=labels, y=cosine_values),
            go.Bar(name='Euclidean Distance', x=labels, y=euclidean_values)
        ])
        # Change the bar mode
        fig.update_layout(barmode='group', title_text='<b>Comparison With NCIC<b>')
        return fig
    
