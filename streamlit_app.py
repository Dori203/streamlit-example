import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


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

"""
# Stable diffusion clusterer!

Enter a word or phrase: Explore how AI art interprets your ideas! This app uses Stable Diffusion to find similar concepts and show different images created from them.
"""

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
MAX_PROMPTS = st.slider("Maximum prompts", 1, 1000, 998)
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
 # print(str(samples_num) + ' prompts')
 # print (str(len(GRID_COUNTER)) + " images removed")

st.button("Run query", key=None, help=None, on_click=run_query)

"""
# Welcome to The code!

"""


# model_types = bert, robeta, gpt2
# model_name = roberta-base, gpt2-medium,
def run_plm():
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

st.button("Run embedding", key=None, help=None, on_click=run_plm)









# Original code

indices = np.linspace(0, 1, num_points)
theta = 2 * np.pi * num_turns * indices
radius = indices

x = radius * np.cos(theta)
y = radius * np.sin(theta)

df = pd.DataFrame({
    "x": x,
    "y": y,
    "idx": indices,
    "rand": np.random.randn(num_points),
})

st.altair_chart(alt.Chart(df, height=700, width=700)
    .mark_point(filled=True)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        color=alt.Color("idx", legend=None, scale=alt.Scale()),
        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
    ))
