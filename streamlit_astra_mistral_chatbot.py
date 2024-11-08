#"vHxFCBV5OZbQYxQrSg0NugMNSdpCTdZ4"
import streamlit as st
import os
import json
from mistralai import Mistral, models
from astrapy import DataAPIClient
import torch
from transformers import BertTokenizer, BertModel
import time

# Mistral AI API configuration
api_key = "vHxFCBV5OZbQYxQrSg0NugMNSdpCTdZ4"
model = "mistral-large-latest"
client_mistral = Mistral(api_key=api_key)

# Astra DB configuration
client_astra = DataAPIClient("AstraCS:ruQuZHsXMkHUYFjfcsXzITOs:b0545554bc378e2b8ede0e41815074a83f4a5e0a3056f76e7154f1ad247fe39f")
db = client_astra.get_database_by_api_endpoint("https://c983bb3f-4942-41ce-bee8-d38ff2fca9b4-us-east1.apps.astra.datastax.com")

# BERT model and tokenizer configuration
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

def refine_user_input(user_input):
    inputs = tokenizer(user_input, return_tensors='pt')
    outputs = bert_model(**inputs)
    pooled_output = outputs.pooler_output
    refined_input = torch.nn.functional.normalize(pooled_output)
    return refined_input.detach().numpy()[0].tolist()

def search_astra_db(query):
    collection_name = "nyaya"
    collection = db.get_collection(collection_name)
    results = collection.find(
        sort={"$vectorize": query},
        limit=2,
        projection={"$vectorize": True},
        include_similarity=True,
    )
    return "\n".join([str(document) for document in results])

def handle_conversation(user_input, prev_response):
    # Refine user input
    refined_input = refine_user_input(user_input)

    # Search Astra DB
    astra_results = search_astra_db(user_input)

    # Prepare chat message
    messages = []

    # Append previous bot response
    if prev_response:
        messages.append({"role": "assistant", "content": prev_response})

    # Append new user input
    messages.append({"role": "user", "content": user_input})

    # Append Astra DB search results as a user message for now
    messages.append({"role": "user", "content": f"Search results: {astra_results}"})

    # Invoke Mistral AI API with rate limiting and exponential backoff
    retry_count = 0
    while retry_count < 5:  # Try up to 5 times
        try:
            chat_response = client_mistral.chat.complete(
                model=model,
                messages=messages,
            )
            break  # If successful, break the retry loop
        except models.SDKError as e:
            if "Requests rate limit exceeded" in str(e):
                st.write(f"Rate limit exceeded. Waiting {2 ** retry_count * 60} seconds...")
                time.sleep(2 ** retry_count * 60)  # Exponential backoff
                retry_count += 1
            else:
                raise

    # Return response
    response = chat_response.choices[0].message.content
    return response

# Streamlit app
st.title("AI Judicial Advisor")

if "prev_response" not in st.session_state:
    st.session_state.prev_response = ""

user_input = st.text_input("You:")

if st.button("Send"):
    if user_input.lower() == "exit":
        st.stop()

    response = handle_conversation(user_input, st.session_state.prev_response)
    st.session_state.prev_response = response
    st.write("Bot:", response)
