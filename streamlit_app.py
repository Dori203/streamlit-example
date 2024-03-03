import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import torch

import pandas as pd
# import plotly.express as px
import numpy as np
# import matplotlib.pyplot as plt
import string
# pip install simpletransformers
# pip install umap-learn
# pip install -U dash
# from dash import jupyter_dash
import requests
import json
# from dash import jupyter_dash
# pip install jupyter_dash
# from jupyter_dash import JupyterDash

# from dash import Dash, dcc, html, Input, Output, no_update
# import plotly.graph_objects as go
# import pandas as pd
from simpletransformers.language_representation import RepresentationModel
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

"""
# Stable diffusion clusterer!

Enter a word or phrase: Explore how AI art interprets your ideas! This app uses Stable Diffusion to find similar concepts and show different images created from them.
"""
st.write(f'**cuda is available:** {torch.cuda.is_available()}')

 # params
# query = 'fish' # @param {type:"string"}
# MAX_PROMPTS = 998 # @param {type:"slider", min:51, max:1000, step:1}
# mapping = 'PLM' # @param ["PLM", "TF-IFD", "Doc2Vec"]
embedding_dimension = 2 # @param {type:"slider", min:2, max:3, step:1}
use_modifiers = True # @param {type:"boolean"}
join_modifiers = False # @param {type:"boolean"}
remove_outliers = True # @param {type:"boolean"}
umap_embedding = False # @param {type:"boolean"}
# cluster_threshold = 0.15 # @param {type:"slider", min:0.05, max:1, step:0.05}
# num_of_clusters = 12 # @param {type:"slider", min:5, max:20, step:1}
# TOP_WORDS =  6 # @param {type:"slider", min:1, max:10, step:1}

query = st.text_input("Query", "Donald Trump")
MAX_PROMPTS = st.slider("Maximum prompts", 1, 1000, 110)
cluster_threshold = st.slider("Cluster threshole", 0.05, 1.00, 0.15)
num_of_clusters = st.slider("number_of_clusters", 5, 20, 12)
TOP_WORDS =  st.slider("number of top words", 1, 10, 6)
num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.slider("Number of turns in spiral", 1, 300, 31)



PROMPT_URI = []
GRID_COUNTER = []

def merge_modifiers(modifier_list):
  if (join_modifiers):
    modifier_list = ['_'.join(s.split()) for s in modifier_list]
  result = " ".join(modifier_list)
  return result

def next_page(api_url):
  new =  api_url[:-1] + str(int(api_url[-1:])+1)
  print(new)
  return new

def fetch_2(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        if data and "prompts" in data and data["prompts"]:
            for result in data["prompts"]:
              if "is_grid" in result["model_parameters"] and result["model_parameters"]["is_grid"] != 1:
                # image_uri = result["generations"][0]["image_uri"]
                image_uri = result["generations"][0]["thumbnail_uri"]
                prompt = result["prompt"]
                modifiers = result["modifiers"]
                if (use_modifiers):
                  prompt = merge_modifiers(modifiers)
                PROMPT_URI.append((prompt, image_uri))
              else:
                GRID_COUNTER.append("uri")
                # print("removed grid image.")
            #fetch next
            if len(PROMPT_URI) <= MAX_PROMPTS -1:
              print("fetching next page")
              fetch_2(next_page(api_url))
    print(len(PROMPT_URI))
    return

def parse_results(results):
  for result in results:
      image_uri = result["generations"][0]["image_uri"]
      prompt = result["prompt"]
      PROMPT_URI.append((prompt, image_uri))
  return

# Function to send REST API request and extract image_uri
def get_responses(prompt):
    # api_url = f"https://devapi.krea.ai/prompts/?format=json&search={prompt}"
    api_url = f"https://search.krea.ai/api/prompts?query={prompt}&pageSize=50&page=1"


    fetch_2(api_url)
    return PROMPT_URI


def run_query():
  result_tuples = get_responses(query)
  df = pd.DataFrame(result_tuples, columns=["prompt", "image_URI"])
  sentences_and_images = pd.DataFrame(result_tuples, columns=["prompt", "image_URI"])
  sentences = df['prompt'].values.tolist()
  image_url = df['image_URI'].values.tolist()
  samples_num = len(sentences)
  st.text("printing stuff")
  st.text(str(samples_num) + " prompts")
  st.text(str(len(GRID_COUNTER)) + " images removed")
  model = RepresentationModel(
          model_type="roberta",
          model_name="roberta-base",
          use_cuda=False
          )
  sentence_vectors = model.encode_sentences(sentences, combine_strategy="mean")
  norm = np.linalg.norm(sentence_vectors, ord=2, axis=1)
  sentence_vactor_normalized = sentence_vectors / norm[:,None]
  sentences_np = np.array(sentences, dtype=object)
  image_urls_np = np.array(image_url, dtype=object)
  meanings_all = pd.DataFrame(sentences_and_images, columns=['prompt', 'image_URI'])
  X_emb = sentence_vactor_normalized
  st.text("finished the embedding")
 
  # Step 1: Detect and remove outliers using Isolation Forest
  if remove_outliers:
    iso_forest = IsolationForest(contamination=0.5)  # Adjust contamination based on your dataset
    outlier_mask = iso_forest.fit_predict(X_emb)
    X_emb = X_emb[outlier_mask == 1]
  
  # Create a random dataset for demonstration
  np.random.seed(42)
  num_clusters = num_of_clusters
  kmeans = KMeans(n_clusters=num_clusters)
  cluster_labels = kmeans.fit_predict(X_emb)
  silhouette_scores = silhouette_score(X_emb, cluster_labels)


  st.text("Get Top X by Centroid Distnace")

  # Get Top X by Centroid Distnace
  cluster_centers = kmeans.cluster_centers_
  distances = np.linalg.norm(X_emb - cluster_centers[cluster_labels], axis=1)
  cluster_selection_mask = distances < cluster_threshold
  if remove_outliers:
    meanings_all = meanings_all[outlier_mask == 1]
    meanings_all = meanings_all[cluster_selection_mask]
    cluster_labels = cluster_labels[cluster_selection_mask]
  
  meanings_all['cluster'] = cluster_labels
  meanings = meanings_all.groupby('cluster').apply(lambda x: " ".join(x['prompt'])).rename('text').reset_index()
  
   # Initialize the TF-IDF vectorizer
  tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=0.15, max_df=0.85)
  
  # Fit and transform the text column
  tfidf_matrix = tfidf_vectorizer.fit_transform(meanings['text'])
  # tfidf_matrix = tfidf_vectorizer.fit_transform(testCorpus)
  
  
  # Get feature names (words)
  feature_names = tfidf_vectorizer.get_feature_names_out()
  
  # Convert TF-IDF matrix to an array
  tfidf_array = tfidf_matrix.toarray()
  
  # Find the top 5 words based on their importance in each document
  top_words_per_document = []
  for doc_tfidf in tfidf_array:
      # Sort indices based on TF-IDF values in descending order
      sorted_indices = (-doc_tfidf).argsort()[:TOP_WORDS]  # Get the indices of the top 5 terms
      top_words = [feature_names[idx] for idx in sorted_indices]
      top_words_per_document.append(top_words)
  
  # Add top words to DataFrame
  meanings['cluster_label'] = top_words_per_document
  meanings['cluster_label'] = meanings['cluster_label'].apply(lambda x: ', '.join(x))
  cluster_description = meanings['cluster_label'].values.tolist()
  df = meanings_all.merge(meanings.drop(columns=['text']), on='cluster')
  grouped = df.groupby('cluster').apply(lambda x: x.to_dict(orient='records')).reset_index()
  # del grouped['cluster']
  grouped.columns = ['cluster', 'data']

  st.text("Done")

st.button("Run query", key=None, help=None, on_click=run_query)

"""
# Welcome to The code!

"""







