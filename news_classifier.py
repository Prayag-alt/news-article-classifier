"""
news_classifier.py
===================
Model loading, prediction, confidence / uncertainty estimation,
and sentiment analysis.

This module wraps the trained Linear SVM and TF-IDF vectorizer so that
the Streamlit UI never touches scikit-learn directly.
"""

import os
import warnings

import numpy as np
import joblib
from textblob import TextBlob

from text_processing import preprocess_text

# Suppress the sklearn version mismatch warning (model trained with 1.6.1,
# runtime may be 1.4.x).  The LinearSVC API is stable between these versions.
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Model & Vectorizer Loading
# ---------------------------------------------------------------------------

# Default file paths (relative to app working directory)
MODEL_PATH = os.environ.get("MODEL_PATH", "news_category_svm.pkl")
TFIDF_PATH = os.environ.get("TFIDF_PATH", "news_category_tfidf.pkl")


def load_model(model_path: str = MODEL_PATH):
    """Load the saved Linear SVM classifier.

    Returns
    -------
    sklearn.svm.LinearSVC or None
        The loaded model, or *None* if the file is missing.
    str or None
        An error message if loading failed.
    """
    if not os.path.isfile(model_path):
        return None, (
            f"Model file not found: `{model_path}`. "
            "Please place `news_category_svm.pkl` in the project root."
        )
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"


def load_tfidf(tfidf_path: str = TFIDF_PATH):
    """Load the saved TF-IDF vectorizer.

    Returns
    -------
    sklearn.feature_extraction.text.TfidfVectorizer or None
        The loaded vectorizer, or *None* if the file is missing.
    str or None
        An error message if loading failed.
    """
    if not os.path.isfile(tfidf_path):
        return None, (
            f"Vectorizer file not found: `{tfidf_path}`. "
            "Please place `news_category_tfidf.pkl` in the project root."
        )
    try:
        tfidf = joblib.load(tfidf_path)
        # Compatibility fix across scikit-learn versions (e.g., 1.4 vs 1.6+)
        if hasattr(tfidf, "_tfidf"):
            if not hasattr(tfidf._tfidf, "_idf_diag") and "idf_" in tfidf._tfidf.__dict__:
                import scipy.sparse as sp
                idf_arr = tfidf._tfidf.__dict__["idf_"]
                tfidf._tfidf._idf_diag = sp.diags(
                    idf_arr, offsets=0, shape=(len(idf_arr), len(idf_arr)), format="csr", dtype=float
                )
        return tfidf, None
    except Exception as exc:
        return None, f"Failed to load vectorizer: {exc}"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_text(text: str, model, tfidf_vectorizer) -> dict:
    """Classify pre-processed *text* and return a rich result dictionary.

    Parameters
    ----------
    text : str
        **Raw** (unprocessed) user input text.
    model : LinearSVC
        The trained classifier.
    tfidf_vectorizer : TfidfVectorizer
        The fitted vectorizer.

    Returns
    -------
    dict
        Keys:
        - ``predicted_category`` (str)
        - ``cleaned_text`` (str)
        - ``decision_scores`` (list[dict]) — top-5 categories with scores
        - ``top_score`` (float)
        - ``margin`` (float) — gap between 1st and 2nd highest scores
        - ``confidence_level`` (str) — "High" / "Moderate" / "Low"
        - ``is_uncertain`` (bool)
        - ``error`` (str or None)
    """
    result = {
        "predicted_category": None,
        "cleaned_text": "",
        "decision_scores": [],
        "top_score": 0.0,
        "margin": 0.0,
        "confidence_level": "Low",
        "is_uncertain": True,
        "error": None,
    }

    if not text or not text.strip():
        result["error"] = "Input text is empty."
        return result

    cleaned = preprocess_text(text)
    result["cleaned_text"] = cleaned

    if not cleaned.strip():
        result["error"] = (
            "After removing stop-words and non-alphabetic characters, "
            "no usable words remain. Please enter more descriptive text."
        )
        return result

    # Vectorise
    X = tfidf_vectorizer.transform([cleaned])

    # Predict
    predicted = model.predict(X)[0]
    result["predicted_category"] = predicted

    # Decision scores for uncertainty estimation
    scores = model.decision_function(X)  # shape: (1, n_classes)

    if scores.ndim == 1:
        # Binary case (unlikely here but safe)
        scores = scores.reshape(1, -1)

    scores_flat = scores.flatten()
    classes = model.classes_

    # Build ranked list
    ranked_indices = np.argsort(scores_flat)[::-1]
    top_n = min(5, len(classes))
    decision_scores = [
        {
            "category": classes[idx],
            "score": round(float(scores_flat[idx]), 4),
        }
        for idx in ranked_indices[:top_n]
    ]
    result["decision_scores"] = decision_scores

    top_score = float(scores_flat[ranked_indices[0]])
    second_score = float(scores_flat[ranked_indices[1]]) if len(ranked_indices) > 1 else 0.0
    margin = top_score - second_score

    result["top_score"] = round(top_score, 4)
    result["margin"] = round(margin, 4)

    # Heuristic confidence level based on the margin between the top two
    # decision scores.  These thresholds are chosen empirically.
    if margin >= 0.4:
        result["confidence_level"] = "High"
        result["is_uncertain"] = False
    elif margin >= 0.15:
        result["confidence_level"] = "Moderate"
        result["is_uncertain"] = False
    else:
        result["confidence_level"] = "Low"
        result["is_uncertain"] = True

    return result


# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------

def analyze_sentiment(text: str) -> dict:
    """Determine the sentiment polarity of *text* using TextBlob.

    TextBlob uses a rule-based / pattern-based approach.  It returns a
    polarity score in [-1.0, +1.0] and a subjectivity score in [0.0, 1.0].

    Parameters
    ----------
    text : str
        Raw (unprocessed) input text.

    Returns
    -------
    dict
        Keys:
        - ``label`` (str) — "Positive" / "Negative" / "Neutral"
        - ``polarity`` (float) — in [-1, 1]
        - ``subjectivity`` (float) — in [0, 1]
    """
    if not text or not text.strip():
        return {"label": "Neutral", "polarity": 0.0, "subjectivity": 0.0}

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label": label,
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
    }
