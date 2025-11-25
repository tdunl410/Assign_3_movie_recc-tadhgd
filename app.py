# app.py
import os
import math
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import requests
import numpy as np
import pandas as pd

import streamlit as st
import altair as alt

# -----------------------------
# Constants & basic config
# -----------------------------

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

# Stable genre IDs from TMDB docs
GENRE_MAP = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}

APP_NAME = "CineSense"


# -----------------------------
# TMDB helpers
# -----------------------------

def get_tmdb_api_key() -> Optional[str]:
    """
    Looks for TMDB API key in:
    1) Streamlit secrets
    2) Environment variable
    3) Sidebar text input
    """
    key = None

    # 1) Streamlit secrets (recommended for deployment)
    try:
        key = st.secrets.get("TMDB_API_KEY", None)
    except Exception:
        key = None

    # 2) Environment variable fallback
    if not key:
        key = os.getenv("TMDB_API_KEY")

    # 3) Sidebar input override
    st.sidebar.subheader("TMDB API Key")
    key_input = st.sidebar.text_input(
        "Enter your TMDB API key",
        type="password",
        help="Create one at themoviedb.org > Settings > API.",
        value=key or "",
    )
    if key_input:
        key = key_input

    return key or None


def tmdb_get(path: str, api_key: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{TMDB_BASE_URL}{path}"
    params = params or {}
    params["api_key"] = api_key
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def search_movies(query: str, api_key: str, page: int = 1) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    data = tmdb_get("/search/movie", api_key, {"query": query, "page": page})
    return data.get("results", [])


def search_people(query: str, api_key: str) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    data = tmdb_get("/search/person", api_key, {"query": query})
    return data.get("results", [])


def get_movie_details(movie_id: int, api_key: str) -> Dict[str, Any]:
    """
    Get movie details with credits, keywords and release dates appended.
    """
    params = {
        "append_to_response": "credits,keywords,release_dates"
    }
    return tmdb_get(f"/movie/{movie_id}", api_key, params=params)


def discover_movies(
    api_key: str,
    *,
    year_range: Optional[List[int]] = None,
    min_rating: float = 0.0,
    min_votes: int = 0,
    runtime_range: Optional[List[int]] = None,
    certification: Optional[str] = None,
    language: Optional[str] = None,
    genre_ids: Optional[List[int]] = None,
    genre_mode: str = "AND",
    people_ids: Optional[List[int]] = None,
    max_pages: int = 2,
) -> List[Dict[str, Any]]:
    """
    Uses /discover/movie with various filters.
    """
    params: Dict[str, Any] = {
        "sort_by": "popularity.desc",
        "include_adult": False,
        "include_video": False,
        "vote_average.gte": min_rating,
        "vote_count.gte": min_votes,
    }

    if year_range:
        params["primary_release_date.gte"] = f"{year_range[0]}-01-01"
        params["primary_release_date.lte"] = f"{year_range[1]}-12-31"

    if runtime_range:
        params["with_runtime.gte"] = runtime_range[0]
        params["with_runtime.lte"] = runtime_range[1]

    if certification:
        params["certification_country"] = "US"
        params["certification.lte"] = certification

    if language:
        params["with_original_language"] = language

    if genre_ids:
        ids_str = [str(g) for g in genre_ids]
        if genre_mode == "AND":
            params["with_genres"] = ",".join(ids_str)
        else:
            params["with_genres"] = "|".join(ids_str)

    if people_ids:
        params["with_people"] = ",".join(str(p) for p in people_ids)

    results: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        try:
            data = tmdb_get("/discover/movie", api_key, params)
            results.extend(data.get("results", []))
        except Exception:
            break
    return results


def get_trending(api_key: str) -> List[Dict[str, Any]]:
    data = tmdb_get("/trending/movie/week", api_key, {})
    return data.get("results", [])


def get_popular(api_key: str, pages: int = 1) -> List[Dict[str, Any]]:
    results = []
    for page in range(1, pages + 1):
        data = tmdb_get("/movie/popular", api_key, {"page": page})
        results.extend(data.get("results", []))
    return results


def get_similar_movies(movie_id: int, api_key: str, pages: int = 1) -> List[Dict[str, Any]]:
    results = []
    for page in range(1, pages + 1):
        data = tmdb_get(f"/movie/{movie_id}/similar", api_key, {"page": page})
        results.extend(data.get("results", []))
    return results


# -----------------------------
# Simple sentiment (no external libs)
# -----------------------------

def compute_sentiment(text: str) -> float:
    """
    Very simple lexicon-based sentiment:
    - counts positive and negative words
    - returns score in roughly [-1, 1]
    This keeps us 'sentiment-based' without external libraries.
    """
    if not text:
        return 0.0

    positive_words = {
        "great", "good", "amazing", "wonderful", "excellent", "fantastic",
        "fun", "enjoyable", "heartwarming", "uplifting", "beautiful",
        "inspiring", "hilarious", "funny", "sweet", "charming"
    }
    negative_words = {
        "bad", "terrible", "awful", "horrible", "boring", "dull",
        "dark", "grim", "depressing", "sad", "tragic", "violent",
        "disturbing", "brutal", "bloody", "gory"
    }

    text_lower = text.lower()
    tokens = text_lower.split()

    pos = sum(1 for t in tokens if t.strip(".,!?;:") in positive_words)
    neg = sum(1 for t in tokens if t.strip(".,!?;:") in negative_words)

    if pos == 0 and neg == 0:
        return 0.0

    score = (pos - neg) / (pos + neg)
    return float(score)


def mood_label(polarity: float) -> str:
    if polarity > 0.15:
        return "feel-good / positive"
    if polarity < -0.15:
        return "dark / serious"
    return "neutral / mixed"


def extract_people(details: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    Returns (cast_names, director_name).
    Correct type hint using Tuple[...] instead of (List[str], str).
    """
    credits = details.get("credits", {})
    cast = credits.get("cast", []) or []
    crew = credits.get("crew", []) or []

    cast_names = [c["name"] for c in cast[:5] if "name" in c]
    director = ""
    for c in crew:
        if c.get("job") == "Director":
            director = c.get("name", "")
            break
    return cast_names, director


def extract_genres(details: Dict[str, Any]) -> List[str]:
    genres = details.get("genres", []) or []
    return [g["name"] for g in genres if "name" in g]


def extract_keywords(details: Dict[str, Any]) -> List[str]:
    if "keywords" in details and isinstance(details["keywords"], dict):
        kw = details["keywords"].get("keywords", [])
    else:
        kw = details.get("keywords", []) or []
    return [k["name"] for k in kw if "name" in k]


def build_feature_frame(
    seed_details: Dict[str, Any],
    candidate_details: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Build a DataFrame with combined_text + sentiment for seed and candidates.
    """
    rows = []
    all_movies = [("SEED", seed_details)] + [("CAND", d) for d in candidate_details]

    for kind, d in all_movies:
        movie_id = d.get("id")
        title = d.get("title") or d.get("name", "Unknown title")
        overview = d.get("overview", "") or ""
        genres = extract_genres(d)
        keywords = extract_keywords(d)
        cast, director = extract_people(d)

        combined_parts = [
            title,
            overview,
            " ".join(genres),
            " ".join(keywords),
            " ".join(cast),
            director or ""
        ]
        combined_text = " ".join([p for p in combined_parts if p])

        sentiment = compute_sentiment(overview)

        rows.append({
            "kind": kind,
            "id": movie_id,
            "title": title,
            "overview": overview,
            "genres": genres,
            "keywords": keywords,
            "cast": cast,
            "director": director,
            "combined_text": combined_text,
            "sentiment": sentiment,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Pure-Python TF-IDF + cosine similarity
# -----------------------------

def tokenize(text: str) -> List[str]:
    return [t.lower().strip(".,!?;:()[]\"'") for t in text.split() if t.strip()]


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    count = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {word: count[word] / total for word in count}


def compute_idf(docs: List[List[str]]) -> Dict[str, float]:
    N = len(docs)
    idf: Dict[str, float] = {}
    all_words = set(word for doc in docs for word in set(doc))
    for word in all_words:
        containing = sum(1 for doc in docs if word in doc)
        idf[word] = math.log((N + 1) / (containing + 1)) + 1.0
    return idf


def compute_tfidf(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = compute_tf(tokens)
    return {word: tf[word] * idf.get(word, 0.0) for word in tf}


def cosine_sim_vec(a: Dict[str, float], b: Dict[str, float]) -> float:
    common = set(a.keys()) & set(b.keys())
    dot = sum(a[w] * b[w] for w in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def hybrid_content_sentiment_recs(
    seed_details: Dict[str, Any],
    candidate_details: List[Dict[str, Any]],
    top_k: int = 10
) -> pd.DataFrame:
    """
    Content-based similarity using pure-Python TF-IDF + sentiment penalty.
    """
    if not candidate_details:
        return pd.DataFrame()

    df = build_feature_frame(seed_details, candidate_details)

    # Tokenize all texts
    docs = [tokenize(t) for t in df["combined_text"]]

    # Compute IDF across all docs
    idf = compute_idf(docs)

    # Compute TF-IDF vectors
    tfidf_vecs = [compute_tfidf(tokens, idf) for tokens in docs]

    # Seed is row 0
    seed_vec = tfidf_vecs[0]

    # Cosine similarity vs all docs
    sim = [cosine_sim_vec(seed_vec, v) for v in tfidf_vecs]

    seed_sent = df.loc[0, "sentiment"]
    sentiment_diff = np.abs(df["sentiment"].values - seed_sent)

    # Hybrid score: higher content similarity and closer sentiment is better
    sim_arr = np.array(sim)
    hybrid_score = 0.75 * sim_arr - 0.25 * sentiment_diff

    df["content_similarity"] = sim_arr
    df["hybrid_score"] = hybrid_score

    # Exclude the seed itself
    recs = df[df["kind"] == "CAND"].copy()
    recs = recs.sort_values("hybrid_score", ascending=False).head(top_k)

    return recs


def aggregate_watchlist_recs(
    watchlist_ids: List[int],
    api_key: str,
    per_movie: int = 10,
) -> pd.DataFrame:
    """
    Simple profile-based recommendation:
    - For each watchlist movie, get TMDB /similar
    - Aggregate and rank by frequency and rating
    """
    seen_ids = set(watchlist_ids)
    candidate_map: Dict[int, Dict[str, Any]] = {}
    counts: Dict[int, int] = {}

    for mid in watchlist_ids:
        try:
            sims = get_similar_movies(mid, api_key, pages=1)
        except Exception:
            continue
        for m in sims:
            cid = m.get("id")
            if not cid or cid in seen_ids:
                continue
            if cid not in candidate_map:
                candidate_map[cid] = m
                counts[cid] = 0
            counts[cid] += 1

    if not candidate_map:
        return pd.DataFrame()

    rows = []
    for cid, m in candidate_map.items():
        rows.append({
            "id": cid,
            "title": m.get("title"),
            "vote_average": m.get("vote_average", 0.0),
            "vote_count": m.get("vote_count", 0),
            "popularity": m.get("popularity", 0.0),
            "freq": counts[cid],
            "poster_path": m.get("poster_path"),
            "overview": m.get("overview", ""),
        })
    df = pd.DataFrame(rows)
    df["score"] = (
        df["freq"] * 1.0
        + df["vote_average"] * 0.3
        + np.log1p(df["vote_count"]) * 0.1
    )
    return df.sort_values("score", ascending=False).head(per_movie)


# -----------------------------
# UI helpers
# -----------------------------

def show_movie_card(
    m: Dict[str, Any],
    api_key: str,
    show_add_button: bool = True,
    key_prefix: str = ""
):
    cols = st.columns([1, 3])
    with cols[0]:
        poster = m.get("poster_path")
        if poster:
            st.image(TMDB_IMG_BASE + poster, use_container_width=True)
    with cols[1]:
        title = m.get("title") or m.get("name", "Unknown title")
        year = (m.get("release_date") or "")[:4]
        rating = m.get("vote_average", 0.0)
        votes = m.get("vote_count", 0)
        runtime = m.get("runtime")
        lang = m.get("original_language", "")

        st.markdown(f"### {title} ({year})")
        st.markdown(f"⭐ {rating:.1f}  ·  {votes} votes  ·  Lang: `{lang}`")
        if runtime:
            st.markdown(f"⏱ Runtime: {runtime} min")

        overview = m.get("overview", "")
        if overview:
            st.caption(overview[:300] + ("..." if len(overview) > 300 else ""))

        if show_add_button:
            btn_key = f"{key_prefix}add_{m.get('id')}"
            clicked = st.button("➕ Add to watchlist", key=btn_key)

            if clicked:
                # Ensure key exists
                if "watchlist" not in st.session_state:
                    st.session_state["watchlist"] = []

                mid = m.get("id")
                if mid is None:
                    st.error("Could not add this movie (missing ID).")
                    return

                # Work on a copy and reassign to avoid any weirdness
                current = list(st.session_state["watchlist"])
                current_ids = [item["id"] for item in current]

                if mid not in current_ids:
                    current.append({"id": mid, "title": title})
                    st.session_state["watchlist"] = current
                    st.success(f"Added **{title}** to watchlist!")
                else:
                    st.info("Already in watchlist.")


def explain_similarity(seed: Dict[str, Any], rec_row: pd.Series):
    seed_genres = set(extract_genres(seed))
    rec_genres = set(rec_row["genres"])
    genre_overlap = seed_genres.intersection(rec_genres)

    seed_cast, seed_dir = extract_people(seed)
    rec_cast = set(rec_row["cast"])
    cast_overlap = set(seed_cast).intersection(rec_cast)
    dir_match = seed_dir and rec_row["director"] and (seed_dir == rec_row["director"])

    seed_sent = compute_sentiment(seed.get("overview", ""))
    rec_sent = rec_row["sentiment"]

    parts = []
    if genre_overlap:
        parts.append("shared genres: " + ", ".join(sorted(genre_overlap)))
    if cast_overlap:
        parts.append("shared cast: " + ", ".join(sorted(list(cast_overlap))[:3]))
    if dir_match:
        parts.append(f"same director: {seed_dir}")
    parts.append(
        f"similar mood (seed: {mood_label(seed_sent)}, rec: {mood_label(rec_sent)})"
    )
    return " · ".join(parts)


# -----------------------------
# Streamlit app
# -----------------------------

def main():
    st.set_page_config(
        page_title=f"{APP_NAME} – Movie Recommender",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"{APP_NAME} 🎬")
    st.caption("Hybrid content + sentiment movie recommender powered by TMDB")

    api_key = get_tmdb_api_key()
    if not api_key:
        st.error("Please provide a TMDB API key in the sidebar to get started.")
        st.stop()

    # Initialize watchlist
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []

    # Sidebar watchlist debug
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Watchlist size:** {len(st.session_state['watchlist'])}")
    # Debug: raw state (you can remove this once you're happy)
    st.sidebar.caption(f"Raw watchlist: {st.session_state['watchlist']}")

    # ------------ Sidebar filters ------------
    st.sidebar.markdown("## Global Filters")

    year_min, year_max = st.sidebar.slider(
        "Release year range",
        min_value=1950,
        max_value=2025,
        value=(2000, 2025),
        step=1,
    )

    min_rating = st.sidebar.slider(
        "Minimum rating (TMDB avg)",
        min_value=0.0,
        max_value=10.0,
        value=6.5,
        step=0.1,
    )
    min_votes = st.sidebar.slider(
        "Minimum vote count",
        min_value=0,
        max_value=5000,
        value=50,
        step=10,
    )

    runtime_min, runtime_max = st.sidebar.slider(
        "Runtime (minutes)",
        min_value=60,
        max_value=210,
        value=(80, 150),
        step=5,
    )

    certification = st.sidebar.selectbox(
        "Max US certification",
        options=["", "G", "PG", "PG-13", "R", "NC-17"],
        index=0,
        help="Uses TMDB certification filters; empty = no filter.",
    )

    language = st.sidebar.selectbox(
        "Original language",
        options=["", "en", "fr", "es", "de", "ja", "ko", "hi"],
        format_func=lambda x: "Any" if x == "" else x,
    )

    st.sidebar.markdown("### Genres")
    genre_names = list(GENRE_MAP.values())
    selected_genre_labels = st.sidebar.multiselect("Select genres", genre_names)
    selected_genre_ids = [
        gid for gid, name in GENRE_MAP.items() if name in selected_genre_labels
    ]
    genre_mode = st.sidebar.radio("Genre logic", ["AND", "OR"], horizontal=True)

    st.sidebar.markdown("### People preferences")
    actor_pref = st.sidebar.text_input("Preferred actor (optional)")
    director_pref = st.sidebar.text_input("Preferred director (optional)")

    people_ids: List[int] = []
    if actor_pref:
        try:
            ppl = search_people(actor_pref, api_key)
            if ppl:
                people_ids.append(ppl[0]["id"])
        except Exception:
            pass
    if director_pref:
        try:
            ppl = search_people(director_pref, api_key)
            if ppl:
                people_ids.append(ppl[0]["id"])
        except Exception:
            pass

    # ------------ Tabs ------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Natural-language Search", "🎛 Browse & Filter", "💾 My Watchlist", "📈 Trends & Insights"]
    )

    # -------- Tab 1: Natural language search --------
    with tab1:
        st.subheader("Natural-language movie search")
        q = st.text_input(
            "Describe what you want",
            placeholder="e.g., 'mind-bending sci-fi with strong female lead' or 'Tom Cruise action' or a title…",
        )
        if q:
            cols = st.columns(2)

            # Left: title/overview search
            with cols[0]:
                st.markdown("#### Title / plot matches")
                try:
                    results = search_movies(q, api_key)
                except Exception as e:
                    st.error(f"Error searching movies: {e}")
                    results = []

                selected_movie_id = None
                if results:
                    for m in results[:8]:
                        title = m.get("title")
                        year = (m.get("release_date") or "")[:4]
                        label = f"{title} ({year})"
                        if st.button(label, key=f"nl_title_{m['id']}"):
                            selected_movie_id = m["id"]
                    st.caption("Click a title to get recommendations.")
                else:
                    st.info("No movies found matching that query.")

            # Right: person-based
            with cols[1]:
                st.markdown("#### People matches (actors / directors)")
                try:
                    ppl = search_people(q, api_key)
                except Exception as e:
                    st.error(f"Error searching people: {e}")
                    ppl = []

                if ppl:
                    person = ppl[0]
                    st.markdown(f"Top person match: **{person.get('name')}**")
                    if st.button(f"Show movies with {person.get('name')}", key="nl_person_btn"):
                        try:
                            movies_by_person = discover_movies(
                                api_key,
                                year_range=[year_min, year_max],
                                min_rating=min_rating,
                                min_votes=min_votes,
                                runtime_range=[runtime_min, runtime_max],
                                certification=certification or None,
                                language=language or None,
                                genre_ids=selected_genre_ids or None,
                                genre_mode=genre_mode,
                                people_ids=[person["id"]],
                                max_pages=2,
                            )
                            st.markdown(f"##### Movies featuring {person.get('name')}")
                            for m in movies_by_person[:10]:
                                show_movie_card(m, api_key, key_prefix="person_")
                        except Exception as e:
                            st.error(f"Error discovering movies by person: {e}")
                else:
                    st.caption("No strong people matches for that query.")

        # Seed-based recs
        st.markdown("---")
        st.markdown("### Get recommendations for a specific movie")
        seed_title = st.text_input(
            "Type a movie title to use as a seed",
            placeholder="e.g., Inception, The Matrix, Barbie",
            key="seed_search",
        )

        if seed_title:
            try:
                seed_results = search_movies(seed_title, api_key)
            except Exception as e:
                st.error(f"Error searching seed movie: {e}")
                seed_results = []

            if seed_results:
                seed_options = {
                    f"{m.get('title')} ({(m.get('release_date') or '')[:4]})": m
                    for m in seed_results[:10]
                }
                choice_label = st.selectbox("Pick the correct movie", list(seed_options.keys()))
                seed_movie = seed_options[choice_label]

                if st.button("Generate similar movies", key="gen_recs"):
                    try:
                        seed_details = get_movie_details(seed_movie["id"], api_key)
                    except Exception as e:
                        st.error(f"Error fetching movie details: {e}")
                        seed_details = None

                    if seed_details:
                        # Candidate pool: TMDB similar + trending + popular
                        candidates = []
                        try:
                            candidates.extend(get_similar_movies(seed_movie["id"], api_key, pages=2))
                        except Exception:
                            pass
                        try:
                            candidates.extend(get_trending(api_key))
                        except Exception:
                            pass
                        try:
                            candidates.extend(get_popular(api_key, pages=1))
                        except Exception:
                            pass

                        # Deduplicate by ID and fetch detail for each candidate
                        seen_ids = set()
                        cand_details = []
                        for c in candidates:
                            cid = c.get("id")
                            if not cid or cid in seen_ids or cid == seed_movie["id"]:
                                continue
                            seen_ids.add(cid)
                            try:
                                cd = get_movie_details(cid, api_key)
                                cand_details.append(cd)
                            except Exception:
                                continue

                        rec_df = hybrid_content_sentiment_recs(seed_details, cand_details, top_k=10)
                        st.markdown("#### Recommended for you")

                        for _, row in rec_df.iterrows():
                            m = {
                                "id": int(row["id"]),
                                "title": row["title"],
                                "overview": row["overview"],
                                "poster_path": None,
                                "vote_average": 0.0,
                                "vote_count": 0,
                                "release_date": seed_movie.get("release_date"),
                            }
                            try:
                                d = get_movie_details(int(row["id"]), api_key)
                                m["poster_path"] = d.get("poster_path")
                                m["vote_average"] = d.get("vote_average", 0.0)
                                m["vote_count"] = d.get("vote_count", 0)
                                m["runtime"] = d.get("runtime", None)
                                m["original_language"] = d.get("original_language", "")
                            except Exception:
                                pass

                            show_movie_card(m, api_key, show_add_button=True, key_prefix="seedrec_")
                            st.caption("Why this recommendation: " + explain_similarity(seed_details, row))
                            st.markdown("---")

            else:
                st.info("No movies found with that title.")

    # -------- Tab 2: Browse & Filter --------
    with tab2:
        st.subheader("Browse & filter movies")

        if st.button("Run discovery search", key="discover_btn"):
            try:
                movies = discover_movies(
                    api_key,
                    year_range=[year_min, year_max],
                    min_rating=min_rating,
                    min_votes=min_votes,
                    runtime_range=[runtime_min, runtime_max],
                    certification=certification or None,
                    language=language or None,
                    genre_ids=selected_genre_ids or None,
                    genre_mode=genre_mode,
                    people_ids=people_ids or None,
                    max_pages=3,
                )
            except Exception as e:
                st.error(f"Error running discovery: {e}")
                movies = []

            if movies:
                st.success(f"Found {len(movies)} matching movies (showing up to 30).")
                for m in movies[:30]:
                    show_movie_card(m, api_key, key_prefix="discover_")
                    st.markdown("---")
            else:
                st.info("No movies found with those filters.")

    # -------- Tab 3: Watchlist --------
    with tab3:
        st.subheader("My Watchlist & Profile-based Recommendations")

        wl = st.session_state.get("watchlist", [])
        if not wl:
            st.info("Your watchlist is empty. Add movies using the '➕ Add to watchlist' buttons.")
        else:
            st.markdown("### Watchlist")
            for item in wl:
                st.markdown(f"- {item['title']} (ID: {item['id']})")

            if st.button("Generate recommendations from my watchlist", key="wl_recs"):
                watch_ids = [x["id"] for x in wl]
                try:
                    recs_df = aggregate_watchlist_recs(watch_ids, api_key, per_movie=10)
                except Exception as e:
                    st.error(f"Error computing watchlist recommendations: {e}")
                    recs_df = pd.DataFrame()

                if recs_df.empty:
                    st.info("Not enough data yet to recommend from your watchlist.")
                else:
                    st.markdown("### Because you liked these…")
                    for _, row in recs_df.iterrows():
                        m = {
                            "id": int(row["id"]),
                            "title": row["title"],
                            "poster_path": row["poster_path"],
                            "overview": row["overview"],
                            "vote_average": row["vote_average"],
                            "vote_count": row["vote_count"],
                        }
                        show_movie_card(m, api_key, key_prefix="wl_")
                        st.markdown("---")

            if st.button("Clear watchlist"):
                st.session_state["watchlist"] = []
                st.success("Watchlist cleared.")

    # -------- Tab 4: Trends & Insights --------
    with tab4:
        st.subheader("Trending & Popular Movies")

        cols = st.columns(2)
        with cols[0]:
            st.markdown("#### Weekly trending")
            try:
                trending = get_trending(api_key)
            except Exception as e:
                st.error(f"Error fetching trending: {e}")
                trending = []

            for m in trending[:10]:
                show_movie_card(m, api_key, show_add_button=True, key_prefix="trend_")
                st.markdown("---")

        with cols[1]:
            st.markdown("#### Popular right now (sample)")
            try:
                popular = get_popular(api_key, pages=1)
            except Exception as e:
                st.error(f"Error fetching popular: {e}")
                popular = []

            for m in popular[:10]:
                show_movie_card(m, api_key, show_add_button=True, key_prefix="popular_")
                st.markdown("---")

        st.markdown("### Data visualization: rating vs popularity")
        if trending:
            df = pd.DataFrame(trending)
            df = df[
                ["title", "vote_average", "vote_count", "popularity"]
            ].dropna()

            chart = (
                alt.Chart(df)
                .mark_circle(size=80)
                .encode(
                    x=alt.X("popularity:Q", title="Popularity"),
                    y=alt.Y("vote_average:Q", title="Average Rating"),
                    tooltip=["title", "popularity", "vote_average", "vote_count"],
                    color=alt.Color("vote_average:Q", scale=alt.Scale(scheme="blues")),
                )
                .properties(
                    width="container",
                    height=400,
                    title="Trending Movies – Rating vs Popularity",
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough trending data to show the chart.")


if __name__ == "__main__":
    main()