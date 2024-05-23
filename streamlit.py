#!/usr/bin/env python
# coding: utf-8

# In[22]:


import streamlit as st
import tensorflow_hub as hub
from sklearn.neighbors import NearestNeighbors


import pandas as pd 
import pickle


# In[14]:


df = pd.read_csv("/Users/himanshukumarsingh/Downloads/MSc clg stuffs/Sem 2/project/ml project/Top_10000_Movies.csv", engine = 'python')

model_path = "/Users/himanshukumarsingh/Downloads/MSc clg stuffs/Sem 2/project/archive"
model = hub.load(model_path)


# In[16]:


df = df[['genre','original_title','overview','original_language']]
df = df.dropna()


# In[17]:


overviews = list(df['overview'])
titles = list(df['original_title'])
languages = list(df['original_language'])


# In[ ]:


# this function outputs the embedding for the given text
def embed(texts):
    return model(texts)

def recommend(choice,text):
    emb = embed([text])
    if choice == 1:
        neighbors = nn_1.kneighbors(emb, return_distance=False)[0]
        return df['original_title'].iloc[neighbors].tolist()
    
    elif choice == 2:
        neighbors = nn_2.kneighbors(emb, return_distance=False)[0]
        return df['original_title'].iloc[neighbors].tolist()
    
    elif choice == 3:
        neighbors = nn_3.kneighbors(emb, return_distance=False)[0]
        return df['original_title'].iloc[neighbors].tolist()


# In[20]:


embeddings_1 = embed(overviews)
embeddings_2 = embed(titles)
embeddings_3 = embed(languages)


# In[23]:


nn_1 = NearestNeighbors(n_neighbors=10)
nn_1.fit(embeddings_1)

nn_2 = NearestNeighbors(n_neighbors=10)
nn_2.fit(embeddings_2)

nn_3 = NearestNeighbors(n_neighbors=10)
nn_3.fit(embeddings_3)


# In[ ]:


st.title("Movie Recommendation System")
    
option = st.selectbox(
    "On what basis do you want the movie recommendation?",
    ("Genre/Keyword","Movie Name","Language"))

if option == 'Genre/Keyword':
    choice = 1
    response = st.text_input('Enter any genre/keyword')

elif option == 'Movie Name':
    choice = 2
    response = st.text_input('Enter any movie name')

elif option == 'Language':
    choice = 3
    response = st.text_input('Enter the first two alphabets of any language')
        
output = recommend(choice, response)
        
if st.button('Get Recommendations'):
    if response:
        st.write("Recommended Movies are:")
        for movie in output:
            st.write(movie)
                
    else:
        st.write("How can I recommend anything if you're not ready to tell me about your mood!😢")
