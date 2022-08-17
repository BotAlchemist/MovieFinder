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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(layout='wide',page_title="MovieFinder")
st.set_option('deprecation.showPyplotGlobalUse', False)



@st.cache(allow_output_mutation=True)
def load_metadata(filename):
    df_movies= pd.read_csv(filename)
    df_movies= df_movies.dropna()
    df_movies["release_year"] =  pd.to_datetime(df_movies["release_date"])
    df_movies['release_year'] = df_movies['release_year'].dt.year
    df_movies['release_year'] = df_movies['release_year'].astype(int)
    return df_movies
df_movies= load_metadata("movies_db.csv")

df_master= pd.read_csv("train_model_master.csv")



@st.cache(allow_output_mutation=True)
def load_spacy_model(model_name):
    nlp= spacy.load(model_name)
    return nlp
    
nlp= load_spacy_model("en_core_web_lg")


@st.cache(allow_output_mutation=True)
def load_tfidf_model(model_name, matrix_name):
    tfidf_vectorizer = pickle.load(open(model_name, 'rb'))
    tfidf_matrix= pickle.load(open(matrix_name, 'rb'))
    return tfidf_vectorizer, tfidf_matrix

@st.cache(allow_output_mutation=True)
def load_train_dataset(filename):
    return(pd.read_csv(filename))

def get_genres(genres):
    genre_names= ""
    genres= genres.replace("'", '"')
    genres= json.loads(genres)
    if len(genres)>0:
        for i in range(len(genres)):
            genre_names= genre_names+ " " + genres[i]['name']
    else:
        genre_names= "NA"
        
    return genre_names


with st.sidebar:
    i_page= option_menu('MovieFinder', ['Home', 'Train model', 'Search'],
                        default_index=0, icons=['house','cpu', 'search' ], menu_icon= 'cast')
if i_page == 'Train model':
    
    
    tab1, tab2= st.tabs(['Train Synopsis', 'Train Keywords'])
    with tab1:
        st.header("Train model using Synopsis of the movies")
        train1, train2= st.columns(2)
        i_lang= train1.selectbox( "Select language",  df_movies['original_language'].unique())
        i_release_year= train2.slider("Select period", df_movies['release_year'].min(),df_movies['release_year'].max(), (df_movies['release_year'].min(), df_movies['release_year'].max())  )
        
        
        df_movies= df_movies[df_movies['original_language']== i_lang]
        df_movies= df_movies[(df_movies['release_year'] >= i_release_year[0]) & (df_movies['release_year'] <=i_release_year[1] )]
        train1.markdown("####           No. of records: "+ str(len(df_movies)))
        
        
        if train2.button("Click here to train model"):
            with st.spinner('Training the model'):
                df_master= df_master[df_master['original_language']== i_lang]
                train_filename= df_master['file_name'].unique()[0]
                model_name= df_master['model_name'].unique()[0]
                matrix_name= df_master['matrix_name'].unique()[0]
                
                
                
                
                all_stopwords = nlp.Defaults.stop_words
                
                overview_processed=[]
                for index, row in df_movies.iterrows():
                    overview= row['overview']
                    genre_names= get_genres(row['genres'])
                    overview= overview + " " + genre_names
                    overview = nlp(overview)
                    overview= " ".join([token.text for token in overview if not token.text in all_stopwords])
                    overview = nlp(overview)
                    overview= " ".join([token.lemma_ for token in overview if not token.is_punct]).lower()
                    overview = re.sub("\s+", " ", overview)
                    overview_processed.append(overview)
                    
                df_movies['overview_processed']= overview_processed

                # Initialize an instance of tf-idf Vectorizer
                tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
                
                # Generate the tf-idf vectors for the corpus
                tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies['overview_processed'])
                
                # saving the model and vector
                
                pickle.dump(tfidf_vectorizer, open(model_name, 'wb'))
                pickle.dump(tfidf_matrix, open(matrix_name, 'wb'))
             
            #df_movies= df_movies.drop("overview_processed", axis=1)
            df_movies.to_csv(train_filename, index=False)
            
            st.success("Model saved successfully")
            
    
        

    with tab2:
        st.header("Placeholder for keyword train")
    
    
if i_page == 'Search':

    i_lang= st.sidebar.selectbox( "Select language",  df_movies['original_language'].unique())
    df_master= df_master[df_master['original_language']== i_lang]
    train_filename= df_master['file_name'].unique()[0]
    model_name= df_master['model_name'].unique()[0]
    matrix_name= df_master['matrix_name'].unique()[0]
    
   
    df_movies_train= load_train_dataset(train_filename)
    
        
    
    user_input= st.text_input("Enter the synopsis of movie you want to search")
    col1, col2= st.columns(2)
    i_release_year= col1.slider("Select period", df_movies_train['release_year'].min(),df_movies_train['release_year'].max(), (df_movies_train['release_year'].min(),df_movies_train['release_year'].max()) )
    i_voter_avg= col1.slider("Select rating", df_movies_train['vote_average'].min(),df_movies_train['vote_average'].max(), (df_movies_train['vote_average'].min(),df_movies_train['vote_average'].max()) )
    
    
    i_top_results= col2.slider("Select Top n results", 1, 10, 5)
    col2.markdown(" ## ")
    if col2.button('Search movies'):
        if len(user_input)> 0:

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
            
            
            tab1, tab2= st.tabs(['Search using tf-idf method', 'Search using Spacy similarity'])
            with tab1:
                tfidf_vectorizer, tfidf_matrix= load_tfidf_model(model_name, matrix_name)
                user_input_matrix= tfidf_vectorizer.transform(df_user['overview_processed'])
                cosine_sim = linear_kernel(tfidf_matrix, user_input_matrix)
                cosine_sim = [j for sub in cosine_sim for j in sub]
                
                df_movies_train['cosine_similarity']= cosine_sim

                df_movies_train= df_movies_train[(df_movies_train['release_year'] >= i_release_year[0]) & (df_movies_train['release_year'] <=i_release_year[1] )]  
                df_movies_train= df_movies_train[(df_movies_train['vote_average'] >= i_voter_avg[0]) & (df_movies_train['vote_average'] <=i_voter_avg[1] )]
                df_movies_train= df_movies_train.sort_values(['cosine_similarity'], ascending=[False]).head(i_top_results)
                
                
                #st.markdown("""---""")
                for index, row in df_movies_train.iterrows():
                    show_similarity= str(round(row['cosine_similarity']*100,2))
                    st.subheader(row['title']+ " - "+ str(row['release_year']) )
                    show_user_rating= round(row['vote_average'])
                    st.markdown("⭐"*show_user_rating)
                    genre_names= get_genres(row['genres'])
                    st.markdown("##### Genre: " + genre_names  )
                    st.markdown("##### Similarity: " +   show_similarity + " %")
                    st.write(row['overview'])
                    st.markdown("""---""")
            
            
            with tab2:
                user_query= nlp(overview_processed[0])
                spacy_similarity=[]
                for index, row in df_movies_train.iterrows():
                    movie_query= nlp(row['overview_processed'])
                    spacy_similarity.append(user_query.similarity(movie_query))
                    
                df_movies_train['spacy_similarity']= spacy_similarity
                
                df_movies_train= df_movies_train[(df_movies_train['release_year'] >= i_release_year[0]) & (df_movies_train['release_year'] <=i_release_year[1] )]  
                df_movies_train= df_movies_train[(df_movies_train['vote_average'] >= i_voter_avg[0]) & (df_movies_train['vote_average'] <=i_voter_avg[1] )]
                df_movies_train= df_movies_train.sort_values(['spacy_similarity'], ascending=[False]).head(i_top_results)
                
                
                #st.markdown("""---""")
                for index, row in df_movies_train.iterrows():
                    show_similarity= str(round(row['spacy_similarity']*100,2))
                    st.subheader(row['title']+ " - "+ str(row['release_year']) )
                    show_user_rating= round(row['vote_average'])
                    st.markdown("⭐"*show_user_rating)
                    genre_names= get_genres(row['genres'])
                    st.markdown("##### Genre: " + genre_names  )
                    st.markdown("##### Similarity: " +   show_similarity + " %")
                    st.write(row['overview'])
                    st.markdown("""---""")
                    
                    
                
                
                

                
        else:
            st.warning("Please write something")
        
    
    