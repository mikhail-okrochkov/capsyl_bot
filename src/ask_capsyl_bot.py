import pandas as pd
import numpy as np
import torch
import clip
from openai import OpenAI
import ast  # assumes correct openai CLIP is installed
from dotenv import load_dotenv
import os
import json
from rapidfuzz import fuzz

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)
model.eval()

df = pd.read_excel("../data/google_photos_metadata_with_location.xlsx")
metadata_embeddings = np.load("../data/metadata_embeddings_l14_336.npy")
image_embeddings = np.load("../data/image_embeddings_l14_336.npy")

load_dotenv()
api_key = os
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

history = {"messages": [], "last_results": None}  # store previous top-k results


def match_names_list(names_list, query_name, threshold=80):
    """
    Returns True if any name in names_list matches query_name with similarity >= threshold
    """
    if not names_list:
        return False
    return any(fuzz.ratio(n.lower(), query_name.lower()) >= threshold for n in names_list)


def embed_text(text):
    text = f"A photo of {text}"
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = model.encode_text(text_tokens)
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def cosine_similarity(query_emb, embeddings):
    return embeddings.dot(query_emb)


def search_photos_gpt(user_message, top_k=10, df_subset=None):
    # 1. Extract structured filters
    filters = parse_user_query(user_message)
    year = filters.get("year")
    month = filters.get("month")
    people = filters.get("people", [])
    location = filters.get("location")
    keywords = filters.get("keywords", [])

    # 2. Determine dataset to search
    df_search = df_subset if df_subset is not None else df.copy()

    # 3. Apply filters
    if year:
        df_search = df_search[df_search["datetime"].dt.year == year]
    if month:
        df_search = df_search[df_search["datetime"].dt.month == month]
    if people:
        df_search = df_search[df_search["names_list"].apply(lambda x: any(p in x for p in people) if x else False)]

    if location:
        loc_lower = location.lower()
        df_search = df_search[df_search["location_name"].str.contains(loc_lower, na=False)]

    if len(df_search) == 0:
        return []

    # 4. Embeddings for filtered set
    indices = df_search.index.to_numpy()
    img_emb_search = image_embeddings[indices]
    meta_emb_search = metadata_embeddings[indices]

    # 5. Embed user query
    q_emb = embed_text(user_message)

    # 6. Compute initial similarity (image + metadata)
    sim_meta = cosine_similarity(q_emb, meta_emb_search)
    sim_img = cosine_similarity(q_emb, img_emb_search)
    sim_combined = (sim_meta + sim_img) / 2.0

    # 7. Incorporate keyword embeddings if present
    if keywords:
        keyword_embs = np.array([embed_text(k) for k in keywords])
        # Average similarity across all keywords
        sim_keywords = np.mean([cosine_similarity(k_emb, meta_emb_search) for k_emb in keyword_embs], axis=0)
        # Combine with existing similarity (weight can be adjusted)
        # sim_combined = sim_keywords
        sim_combined = (sim_combined + sim_keywords) / 2.0

    # 8. Top-k results
    idx_sorted = np.argsort(-sim_combined)[:top_k]
    results = []
    for rank, idx in enumerate(idx_sorted):
        row = df_search.iloc[idx].to_dict()
        row["_score"] = float(sim_combined[idx])
        row["_rank"] = int(rank + 1)
        results.append(row)

    return results


def parse_user_query(user_message):
    system_prompt = """
You are an assistant that extracts structured search filters from a user's natural language query.
Always respond with a JSON object with the following keys:
- year (int or null)
- month (int 1-12 or null)
- people (list of strings)
- location (string or null)
- keywords (list of strings): include all keywords explicitly mentioned by the user, 
  and also append any relevant synonyms or related terms that could match similar photos 
  (e.g., "sunset" → ["sunset", "golden hour", "dusk"]).

Be concise and only return valid JSON, without extra explanation.
"""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

    response = client.chat.completions.create(model="gpt-5", messages=messages)

    text = response.choices[0].message.content

    try:
        filters = json.loads(text)
    except Exception:
        filters = {}

    return filters


def chat_with_photos(user_message, top_k=5, use_previous_results=True):
    # Decide which dataset to search
    df_subset = None
    if use_previous_results and history["last_results"]:
        df_subset = pd.DataFrame(history["last_results"])

    # Search photos
    results = search_photos_gpt(user_message, top_k=top_k, df_subset=df_subset)

    # Build system prompt for GPT-5
    system_prompt = f"""
    You are a helpful photo assistant.
    Here are the top {top_k} candidate photos from the user's library:
    {results}
    Respond naturally to the user's question and reference relevant photo names.
    """

    messages = history["messages"] + [{"role": "user", "content": user_message}]
    messages = [{"role": "system", "content": system_prompt}] + messages

    response = client.chat.completions.create(model="gpt-5", messages=messages)

    bot_message = response.choices[0].message.content

    # Update conversation history
    history["messages"].append({"role": "user", "content": user_message})
    history["messages"].append({"role": "assistant", "content": bot_message})
    history["last_results"] = results  # save for next query

    return bot_message, results
