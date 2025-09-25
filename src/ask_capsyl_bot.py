import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel("../data/google_photos_metadata_with_location.xlsx")
caption_embeddings = np.load("../data/caption_embeddings.npy")

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


def embed_text_openai(text: str):
    """
    Embed text using text-embedding-3-large
    """
    resp = client.embeddings.create(model="text-embedding-3-large", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def apply_structured_filters(df, filters):
    """
    Apply structured positive and negative filters to a photo DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns 'datetime', 'names_list', 'location_name'.
        year, month: int or None for positive filters.
        exclude_years, exclude_months: list of int for negative filters.
        people, exclude_people: list of strings.
        location: string for positive filter.
        exclude_locations: list of strings for negative filter.

    Returns:
        pd.DataFrame: filtered DataFrame
    """
    df_search = df.copy()

    year = filters.get("year")
    exclude_years = filters.get("exclude_years", [])
    month = filters.get("month")
    exclude_months = filters.get("exclude_months", [])
    people = filters.get("people", [])
    exclude_people = filters.get("exclude_people", [])
    location = filters.get("location")
    exclude_locations = filters.get("exclude_locations", [])

    # --- Year ---
    if year:
        df_search = df_search[df_search["datetime"].dt.year == year]
    if exclude_years:
        df_search = df_search[~df_search["datetime"].dt.year.isin(exclude_years)]

    # --- Month ---
    if month:
        df_search = df_search[df_search["datetime"].dt.month == month]
    if exclude_months:
        df_search = df_search[~df_search["datetime"].dt.month.isin(exclude_months)]

    # --- People ---
    if people:
        people_lower = [p.lower() for p in people]
        df_search = df_search[
            df_search["names_list"].apply(
                lambda x: any(p in [n.lower() for n in x] for p in people_lower) if x else False
            )
        ]
    if exclude_people:
        exclude_people_lower = [p.lower() for p in exclude_people]
        df_search = df_search[
            df_search["names_list"].apply(
                lambda x: not any(p in [n.lower() for n in x] for p in exclude_people_lower) if x else True
            )
        ]

    # --- Location ---
    if location:
        loc_lower = location.lower()
        df_search = df_search[df_search["location_name"].str.contains(loc_lower, na=False)]
    if exclude_locations:
        for excl in exclude_locations:
            excl_lower = excl.lower()
            df_search = df_search[~df_search["location_name"].str.contains(excl_lower, na=False)]

    return df_search


def search_photos_gpt(user_message, top_k=10, df_subset=None):
    """
    Search photos using GPT-5 keyword/phrase extraction.
    Only keyword/phrase embeddings are used for similarity search.
    """
    # 1. Extract structured filters (keywords are now phrases)
    filters = parse_user_query(user_message)  # returns structured fields + 3–5 word phrases
    print(filters)

    keywords = filters.get("keywords", [])  # list of 3–5 word phrases

    # 2. Determine dataset to search
    df_search = df_subset if df_subset is not None else df.copy()

    # 3. Apply structured filters
    df_search = apply_structured_filters(df_search, filters)

    # 4. Load precomputed caption embeddings for filtered set

    indices = df_search.index.to_numpy()
    caption_emb_search = np.stack([caption_embeddings[i] for i in indices])

    # 5. Embed GPT-5 keyword phrases and compute similarity
    if keywords:
        keyword_embs = np.stack([embed_text_openai(k) for k in keywords])  # (num_phrases, dim)
        sim_keywords = cosine_similarity(keyword_embs, caption_emb_search)  # (num_phrases, num_images)
        sim_combined = sim_keywords.mean(axis=0)  # average across all phrases -> shape (num_images,)
    else:
        sim_combined = np.zeros(len(df_search))

    # 6. Top-k results
    cutoff = 0.30  # adjust based on experiments
    idx_sorted = np.argsort(-sim_combined)
    results = []

    for rank, idx in enumerate(idx_sorted[:top_k]):
        score = float(sim_combined[idx])
        if score < cutoff:
            break  # stop if we’re below cutoff

        row = df_search.iloc[idx].to_dict()
        row["_score"] = score
        row["_rank"] = int(rank + 1)
        results.append(row)

    return results


def parse_user_query(user_message):
    system_prompt = """
You are a query parser for a photo search system.  
Your job is to take a natural language user request (e.g. "find photos of sunset not in Hawaii with Alice in 2021") 
and return a structured JSON object.  

Follow these rules:  
- Always return valid JSON.  
- Never include explanations or text outside the JSON.  
- Include both positive and negative filters.  
- Values should be simple strings or integers where possible.  
- If no value is specified, return null or an empty list.  

The schema is:  

{
  "year": null or integer,
  "exclude_years": [],
  "month": null or integer,
  "exclude_months": [],
  "people": [],
  "exclude_people": [],
  "location": null or string,
  "exclude_locations": [],
  "keywords": []
}

Examples:

Input: find photos of sunset not in Hawaii  
Output:
{
  "year": null,
  "exclude_years": [],
  "month": null,
  "exclude_months": [],
  "people": [],
  "exclude_people": [],
  "location": null,
  "exclude_locations": ["Hawaii"],
  "keywords": ["sunset"]
}

Input: pictures from 2021 in Paris without Alice  
Output:
{
  "year": 2021,
  "exclude_years": [],
  "month": null,
  "exclude_months": [],
  "people": [],
  "exclude_people": ["Alice"],
  "location": "Paris",
  "exclude_locations": [],
  "keywords": []
}

Input: sunsets from 2020 or 2021, but not in January or February  
Output:
{
  "year": null,
  "exclude_years": [],
  "month": null,
  "exclude_months": [1, 2],
  "people": [],
  "exclude_people": [],
  "location": null,
  "exclude_locations": [],
  "keywords": ["sunset"]
}
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
