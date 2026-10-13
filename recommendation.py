"""
recommendation.py
==================
Personalised news recommendation engine with two modes:

1. **Interest-based feed** — articles filtered by user-selected categories.
2. **Research-to-news** — cosine-similarity ranking between an uploaded
   research text and the news article corpus.

Both modes retrieve real articles from the News Category Dataset.
"""

import os
import json
import random
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from text_processing import preprocess_text, extract_keywords

# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

DATASET_FILENAME = "News_Category_Dataset_v3.json"
DATASET_GZ_FILENAME = "News_Category_Dataset_v3.json.gz"

# Search order for the dataset file
DATASET_SEARCH_PATHS = [
    DATASET_GZ_FILENAME,                                    # compressed in project root
    DATASET_FILENAME,                                       # project root
    os.path.join("..", DATASET_GZ_FILENAME),                 # one level up (gz)
    os.path.join("..", DATASET_FILENAME),                    # one level up
    os.path.join("..", "..", DATASET_GZ_FILENAME),           # two levels up (gz)
    os.path.join("..", "..", DATASET_FILENAME),              # two levels up
    os.path.join("data", DATASET_GZ_FILENAME),              # data subfolder (gz)
    os.path.join("data", DATASET_FILENAME),                 # data subfolder
]


def find_dataset_path() -> Optional[str]:
    """Locate the dataset file (.json or .json.gz) by checking common locations."""
    for path in DATASET_SEARCH_PATHS:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def load_dataset(path: Optional[str] = None) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load the News Category Dataset (JSON-lines format or gzip compressed).

    Parameters
    ----------
    path : str or None
        Explicit path; if *None*, the function searches common locations.

    Returns
    -------
    tuple[pd.DataFrame or None, str or None]
        The loaded DataFrame, or (*None*, error_message).
    """
    if path is None:
        path = find_dataset_path()

    if path is None or not os.path.isfile(path):
        return None, (
            f"Dataset file `{DATASET_FILENAME}` (or `{DATASET_GZ_FILENAME}`) not found. "
            "Please place it in the project root or a `data/` subfolder. "
            "The dataset can be downloaded from Kaggle: "
            "https://www.kaggle.com/datasets/rmisra/news-category-dataset"
        )

    try:
        # Load directly with pandas supporting both plain json and gzip
        if path.endswith(".gz"):
            df = pd.read_json(path, lines=True, compression="gzip")
        else:
            df = pd.read_json(path, lines=True)

        # Ensure expected columns exist
        required_cols = {"headline", "category", "short_description"}
        if not required_cols.issubset(set(df.columns)):
            return None, (
                f"Dataset is missing required columns. "
                f"Expected: {required_cols}. Found: {set(df.columns)}."
            )

        # Combine headline and description for richer text representation
        df["combined_text"] = (
            df["headline"].fillna("") + " " + df["short_description"].fillna("")
        )
        df["combined_text"] = df["combined_text"].str.strip()

        # Clean the date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        return df, None

    except Exception as exc:
        return None, f"Failed to load dataset: {exc}"


# ---------------------------------------------------------------------------
# Interest-Based Feed (Mode A)
# ---------------------------------------------------------------------------

def get_interest_based_feed(
    df: pd.DataFrame,
    selected_categories: list[str],
    n_articles: int = 20,
    sort_by_date: bool = True,
) -> pd.DataFrame:
    """Return articles from the selected categories.

    Parameters
    ----------
    df : pd.DataFrame
        The full news dataset.
    selected_categories : list[str]
        Category names chosen by the user.
    n_articles : int
        Maximum number of articles to return.
    sort_by_date : bool
        If True, return the most recent articles first.

    Returns
    -------
    pd.DataFrame
        Subset of articles with columns preserved.
    """
    if not selected_categories:
        return pd.DataFrame()

    mask = df["category"].isin(selected_categories)
    filtered = df.loc[mask].copy()

    if filtered.empty:
        return filtered

    if sort_by_date and "date" in filtered.columns:
        filtered = filtered.sort_values("date", ascending=False, na_position="last")

    # Take a balanced sample if there are many more articles than requested
    if len(filtered) > n_articles:
        # Stratified sampling: roughly equal articles per category
        per_cat = max(1, n_articles // len(selected_categories))
        samples = []
        for cat in selected_categories:
            cat_df = filtered[filtered["category"] == cat]
            samples.append(cat_df.head(per_cat))
        result = pd.concat(samples, ignore_index=True)

        # Fill remaining slots from the full filtered set
        if len(result) < n_articles:
            remaining_ids = set(filtered.index) - set(result.index)
            extra = filtered.loc[list(remaining_ids)].head(n_articles - len(result))
            result = pd.concat([result, extra], ignore_index=True)

        return result.head(n_articles)

    return filtered.head(n_articles)


# ---------------------------------------------------------------------------
# Research-to-News Recommendations (Mode B)
# ---------------------------------------------------------------------------

def build_corpus_vectors(df: pd.DataFrame, tfidf_vectorizer) -> np.ndarray:
    """Pre-compute TF-IDF vectors for the entire news corpus.

    Uses the **same** fitted vectorizer as the classifier to transform
    the combined headline + description text.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``combined_text`` column.
    tfidf_vectorizer : TfidfVectorizer
        The fitted project vectorizer.

    Returns
    -------
    scipy.sparse matrix
        TF-IDF matrix of shape (n_articles, n_features).
    """
    # Pre-process each article the same way as during training
    cleaned_texts = df["combined_text"].apply(preprocess_text)
    return tfidf_vectorizer.transform(cleaned_texts)


def recommend_from_research(
    research_text: str,
    df: pd.DataFrame,
    corpus_vectors,
    tfidf_vectorizer,
    top_n: int = 15,
) -> tuple[list[dict], list[dict]]:
    """Find news articles most similar to the research text.

    Parameters
    ----------
    research_text : str
        Raw research text (abstract, excerpt, etc.).
    df : pd.DataFrame
        The news dataset with a ``combined_text`` column.
    corpus_vectors : sparse matrix
        Pre-computed TF-IDF vectors for all articles.
    tfidf_vectorizer : TfidfVectorizer
        The fitted vectorizer.
    top_n : int
        Number of recommendations to return.

    Returns
    -------
    tuple[list[dict], list[dict]]
        - A list of recommendation dicts, each containing article metadata
          and a similarity score.
        - A list of keyword dicts extracted from the research text.
    """
    # Pre-process and vectorise the research text
    cleaned = preprocess_text(research_text)
    if not cleaned.strip():
        return [], []

    research_vector = tfidf_vectorizer.transform([cleaned])

    # Extract keywords from the research text
    keywords = extract_keywords(cleaned, tfidf_vectorizer, top_n=12)

    # Compute cosine similarity against every article
    similarities = cosine_similarity(research_vector, corpus_vectors).flatten()

    # Rank by similarity
    top_indices = np.argsort(similarities)[::-1][:top_n]

    recommendations = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score <= 0:
            continue  # skip zero-similarity articles

        row = df.iloc[idx]

        # Build a short explanation based on overlapping keywords
        article_cleaned = preprocess_text(row["combined_text"])
        article_keywords = extract_keywords(
            article_cleaned, tfidf_vectorizer, top_n=8
        )
        article_kw_set = {kw["keyword"] for kw in article_keywords}
        research_kw_set = {kw["keyword"] for kw in keywords}
        overlap = article_kw_set & research_kw_set

        if overlap:
            match_reason = f"Shared terms: {', '.join(sorted(overlap)[:5])}"
        else:
            match_reason = "Matched on overall topic similarity"

        rec = {
            "headline": row.get("headline", ""),
            "category": row.get("category", ""),
            "short_description": row.get("short_description", ""),
            "date": str(row.get("date", ""))[:10] if pd.notna(row.get("date")) else "N/A",
            "authors": row.get("authors", ""),
            "similarity_score": round(score, 4),
            "match_reason": match_reason,
        }
        recommendations.append(rec)

    return recommendations, keywords
