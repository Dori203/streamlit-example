import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import string
!pip install simpletransformers
!pip install umap-learn
!pip install -U dash
from dash import jupyter_dash
import requests
import json
from dash import jupyter_dash
!pip install jupyter_dash
from jupyter_dash import JupyterDash

from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
import pandas as pd

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

// params
query = 'fish' # @param {type:"string"}
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

MAX_PROMPTS = st.slider("Maximum prompts", 1, 1000, 998)
cluster_threshold = st.slider("Cluster threshole", 0.05, 1, 0.15)
num_of_clusters = st.slider("number_of_clusters", 5, 20, 12)
TOP_WORDS =  st.slider("number of top words", 1, 10, 6)
num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

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
