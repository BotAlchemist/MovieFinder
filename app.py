# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 10:57:12 2022

@author: Sumit
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import json
import re
import pickle
import spacy
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(layout='wide',page_title="MovieFinder")
st.set_option('deprecation.showPyplotGlobalUse', False)



@st.cache(allow_output_mutation=True)
def load_metadata(filename):
    df_movies= pd.read_csv(filename)
    return df_movies
df_movies= load_metadata("movies_metedata_filtered.csv")
# df_movies= df_movies[df_movies['original_language']=='en']
# final_columns=['imdb_id', 'title', 'overview', 'genres','release_date']
# df_movies= df_movies[final_columns].dropna()
# df_movies["release_year"] =  pd.to_datetime(df_movies["release_date"], format="%Y-%m-%d")
# df_movies['release_year'] = df_movies['release_year'].dt.year
# df_movies['release_year'] = df_movies['release_year'].astype(int)
# df_movies.to_csv("movies_metedata_filtered.csv", index=False)



@st.cache(allow_output_mutation=True)
def load_spacy_model(model_name):
    nlp= spacy.load(model_name)
    return nlp
    
nlp= load_spacy_model("en_core_web_lg")


with st.sidebar:
    i_page= option_menu('MovieFinder', ['Home', 'Search'],
                        default_index=0, icons=['house', 'search' ], menu_icon= 'cast')
    
if i_page == 'Search':
    user_input= st.text_input("Enter the synopsis of movie")
    i_release_year= st.slider("Select period", df_movies['release_year'].min(),df_movies['release_year'].max(), (1980, 2019) )
    if st.button('Find'):
        
        
        @st.cache(allow_output_mutation=True)
        def load_tfidf_model(model_name, matrix_name):
            tfidf_vectorizer = pickle.load(open(model_name, 'rb'))
            tfidf_matrix= pickle.load(open(matrix_name, 'rb'))
            return tfidf_vectorizer, tfidf_matrix
    
        df_user= pd.DataFrame([user_input], columns=['overview'])
        
        
        all_stopwords = nlp.Defaults.stop_words
        overview_processed=[]
        for index, row in df_user.iterrows():
            overview= row['overview']
            overview = nlp(overview)  
            overview= " ".join([token.text for token in overview if not token.text in all_stopwords])
            
            overview = nlp(overview)
            overview= " ".join([token.lemma_ for token in overview if not token.is_punct]).lower()
            
            overview = re.sub("\s+", " ", overview)
            overview_processed.append(overview)
        df_user['overview_processed']= overview_processed
        
        tfidf_vectorizer, tfidf_matrix= load_tfidf_model("train_model.sav", "train_matrix.sav")
        
        user_input_matrix= tfidf_vectorizer.transform(df_user['overview_processed'])
        cosine_sim = linear_kernel(tfidf_matrix, user_input_matrix)
        cosine_sim = [j for sub in cosine_sim for j in sub]
        
        df_movies['cosine_similarity']= cosine_sim
        
        
        df_movies= df_movies[(df_movies['release_year'] >= i_release_year[0]) & (df_movies['release_year'] <=i_release_year[1] )]
        df_movies= df_movies.sort_values(['cosine_similarity'], ascending=[False]).head(5)
        
        
        
        for index, row in df_movies.iterrows():
            st.subheader(row['title']+ " - "+ str(row['release_year']))
            st.write(row['overview'])
        
    
    