import streamlit as st
import pandas as pd
import numpy as np
import torch
# Allowed to make changes.
import h5py
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    #IR DL model config
    dl_ir_model_name = "sentence-transformers/all-mpnet-base-v2"
    #vanilla model
    dl_ir_model = SentenceTransformer(dl_ir_model_name)
    return dl_ir_model

model = load_model()

@st.cache_data
def load_data():
    #load data
    data = pd.read_csv("clothing_data.csv", encoding="utf-8")
    return data

data = load_data()

@st.cache_resource
def load_embedding():
    #load embeddings
    clothingH5Obj = h5py.File('product_embedding.h5', 'r')
    clothingEmbeddings = clothingH5Obj['embedding']
    return clothingEmbeddings

embedding = load_embedding()

def getQueryEmbedding(query, model):
    queryEmbedding = model.encode(query)
    return queryEmbedding

def getTopK(query, sentEmbeddings, model, k=5):
    queryEmbedding = getQueryEmbedding(query, model)
    topK = torch.topk(util.cos_sim(queryEmbedding, sentEmbeddings), k)
    return topK

query = st.text_input("Product text")
if query != "":
    topKResearchInterests = getTopK([query], np.array(embedding), model, k=5)
    topKIndices = list(topKResearchInterests.indices.numpy()[0])
    topKCosSim = list(topKResearchInterests.values.numpy()[0])
    for i in range(len(topKIndices)):
        st.write(f"The product with rank {i+1} is **{data.iloc[topKIndices[i], 0]}** ([link](%s))"% data.iloc[topKIndices[i], 1])
else:
    st.info("Enter the product text above")



