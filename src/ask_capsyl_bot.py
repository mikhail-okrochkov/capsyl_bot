import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import ast

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


def apply_structured_filters(df_search, filters):
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
    years = filters.get("years", [])
    exclude_years = filters.get("exclude_years", [])
    months = filters.get("months", [])
    exclude_months = filters.get("exclude_months", [])
    people = filters.get("people", [])
    exclude_people = filters.get("exclude_people", [])
    locations = filters.get("locations", [])
    exclude_locations = filters.get("exclude_locations", [])

    df_search["names_list"] = df_search["names_list"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    if years:
        df_search = df_search[df_search["datetime"].dt.year.isin(years)]

    if months:
        df_search = df_search[df_search["datetime"].dt.month.isin(months)]

    if exclude_years:
        df_search = df_search[~df_search["datetime"].dt.year.isin(exclude_years)]

    if exclude_months:
        df_search = df_search[~df_search["datetime"].dt.month.isin(exclude_months)]

    if people:
        df_search = df_search[
            df_search["names_list"].apply(
                lambda x: all(p.lower() in (n.lower() for n in x) for p in people) if x else False
            )
        ]

    if locations:
        df_search = df_search[
            df_search["location_name"].apply(
                lambda loc: any(l.lower() in loc.lower() for l in locations) if pd.notna(loc) else False
            )
        ]
    if exclude_people:
        df_search = df_search[
            ~df_search["names_list"].apply(
                lambda x: any(p.lower() in (n.lower() for n in x) for p in exclude_people) if x else False
            )
        ]

    if exclude_locations:
        df_search = df_search[
            ~df_search["location_name"].apply(
                lambda loc: any(l.lower() in loc.lower() for l in exclude_locations) if pd.notna(loc) else False
            )
        ]

    return df_search


def search_photos_gpt(filters, sim_threshold, top_k=10, df_subset=None):
    """
    Search photos using GPT-5 keyword/phrase extraction.
    Only keyword/phrase embeddings are used for similarity search.
    """

    keywords = filters.get("keywords", [])  # list of 3–5 word phrases

    # 2. Determine dataset to search
    df_search = df_subset if df_subset is not None else df.copy()

    # 3. Apply structured filters
    df_search = apply_structured_filters(df_search, filters)
    print(df_search)
    # 4. Load precomputed caption embeddings for filtered set

    indices = df_search.index.to_numpy()
    caption_emb_search = np.stack([caption_embeddings[i] for i in indices])

    # 5. Embed GPT-5 keyword phrases and compute similarity
    if keywords:
        keyword_embs = np.stack([embed_text_openai(k) for k in keywords])  # (num_phrases, dim)
        sim_keywords = cosine_similarity(keyword_embs, caption_emb_search)  # (num_phrases, num_images)
        sim_combined = sim_keywords.mean(axis=0)  # average across all phrases -> shape (num_images,)

        # Top-k by similarity
        idx_sorted = np.argsort(-sim_combined)
        results = []
        for rank, idx in enumerate(idx_sorted[:top_k]):
            score = float(sim_combined[idx])
            if score < sim_threshold:
                break
            row = df_search.iloc[idx].to_dict()
            row["_score"] = score
            row["_rank"] = int(rank + 1)
            results.append(row)

    else:
        # No keywords → sort by datetime descending
        df_sorted = df_search.sort_values("datetime", ascending=False)
        results = []
        for rank, (_, row) in enumerate(df_sorted.head(top_k).iterrows()):
            row = row.to_dict()
            row["_score"] = None  # no semantic score
            row["_rank"] = int(rank + 1)
            results.append(row)

    return results


def parse_user_query(user_message):
    system_prompt = """
You are an assistant that helps users search a large photo library using natural language queries.

Your tasks:
1. Parse the query into structured filters:
   - years (e.g., "2018")
   - months (e.g., "July")
   - people (names, always lowercase for matching)
   - locations (place names, lowercase for matching)
   - exclude_years
   - exclude_people
   - exclude_locations
   - keywords (list of short descriptive phrases, 3 to 5 words each): 
        - Include all concepts explicitly mentioned by the user,
          as well as relevant synonyms or related terms that could match similar photos.
        - Only include keywords if a certain concept is found, leave empty if simply asking for a photo of a person or a place.
        - Each phrase should be concise, descriptive, and written in natural language suitable for embedding-based search.
        - DO NOT include any person names or location names in the keywords list.
        - Keywords should focus only on visual or descriptive concepts.

2. Decide whether the previous search result can be reused based on the user's query.  
   Add a boolean field `"reuse_previous_result"`:
       - True if the user likely wants to use the previous search result
       - False otherwise

3. Return results in structured JSON format with the following fields:
{
   "years": [],
   "months": [],
   "people": [],
   "locations": [],
   "exclude_years": [],
   "exclude_people": [],
   "exclude_locations": [],
   "keywords": [],
   "reuse_previous_result": false
}

Be concise and deterministic — no free-form explanations, just structured JSON output.
"""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

    response = client.chat.completions.create(model="gpt-5", messages=messages)

    text = response.choices[0].message.content

    try:
        filters = json.loads(text)
    except Exception:
        filters = {}

    return filters


def chat_with_photos(user_message, sim_threshold, top_k=5, use_previous_results=True):
    # Decide which dataset to search
    filters = parse_user_query(user_message)
    print(filters)
    df_subset = None
    use_previous_results = filters.get("reuse_previous_result", [])
    if use_previous_results and history["last_results"]:
        df_subset = pd.DataFrame(history["last_results"])

    # Search photos
    results = search_photos_gpt(filters, sim_threshold, top_k=top_k, df_subset=df_subset)

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


def chat_with_photos_2(user_message, sim_threshold, top_k=20):
    # Decide which dataset to search

    filters = parse_user_query(user_message)
    print(filters)
    df_subset = None
    use_previous_results = filters.get("reuse_previous_result", [])
    if use_previous_results and history["last_results"]:
        df_subset = pd.DataFrame(history["last_results"])

    # Step 1: coarse filter with embeddings
    candidate_results = search_photos_gpt_2(filters, sim_threshold, top_k=top_k, df_subset=df_subset)

    # Step 2: format metadata for GPT refinement
    candidate_metadata = []
    for r in candidate_results:
        candidate_metadata.append(
            {
                "photo_name": r.get("photo_name"),
                "caption": r.get("caption", ""),
                "people": r.get("names_list", []),
                "location": r.get("location_name", ""),
                "date": str(r.get("datetime")),
            }
        )

    # Step 3: LLM refinement
    refine_prompt = f"""
    The user asked: "{user_message}"

    Here are candidate photos with metadata:
    {json.dumps(candidate_metadata, indent=2)}

    From these candidates, select ONLY the photos that best match the user's request.
    Return a JSON list of photo_name values, ordered by relevance.
    """

    refine_messages = [
        {"role": "system", "content": "You are a photo retrieval assistant."},
        {"role": "user", "content": refine_prompt},
    ]
    refine_response = client.chat.completions.create(model="gpt-5", messages=refine_messages)

    try:
        selected_photos = json.loads(refine_response.choices[0].message.content)
    except Exception:
        selected_photos = [r.get("photo_name") for r in candidate_results]  # fallback

    # Step 4: keep only selected results
    final_results = [r for r in candidate_results if r["photo_name"] in selected_photos]

    # Step 5: build assistant reply
    system_prompt = f"""
    You are a helpful photo assistant.
    The user asked: "{user_message}"
    Here are the final selected photos: {final_results}
    Respond naturally and reference relevant photo names.
    """
    messages = history["messages"] + [{"role": "user", "content": user_message}]
    messages = [{"role": "system", "content": system_prompt}] + messages

    response = client.chat.completions.create(model="gpt-5", messages=messages)
    bot_message = response.choices[0].message.content

    # Update history
    history["messages"].append({"role": "user", "content": user_message})
    history["messages"].append({"role": "assistant", "content": bot_message})
    history["last_results"] = final_results

    return bot_message, final_results


def search_photos_gpt_2(filters, sim_threshold, top_k=10, df_subset=None):
    """
    Search photos using GPT-5 keyword/phrase extraction.
    Only keyword/phrase embeddings are used for similarity search.
    """

    keywords = filters.get("keywords", [])  # list of 3–5 word phrases

    # 2. Determine dataset to search
    df_search = df_subset if df_subset is not None else df.copy()

    # 3. Apply structured filters
    df_search = apply_structured_filters(df_search, filters)

    results = []
    for rank, (_, row) in enumerate(df_search.head(top_k).iterrows(), start=1):
        row_dict = row.to_dict()
        row_dict["_score"] = 1.0  # placeholder since no similarity
        row_dict["_rank"] = rank
        results.append(row_dict)

    return results
