"""
text_processing.py
==================
Shared text preprocessing, keyword extraction, and document text extraction.

This module centralises all text transformations so that the classifier,
recommendation engine, and UI layer use exactly the same pipeline.
"""

import re
import nltk

# Download required NLTK data (silently, with fallback)
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
        "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
        "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself",
        "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
        "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
        "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
        "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "why", "with", "you", "your", "yours", "yourself", "yourselves"
    }

# Additional stop-words that frequently appear in news but carry little
# discriminative value for classification or keyword extraction.
EXTRA_STOP_WORDS = {
    "said", "also", "would", "could", "one", "new", "like", "get",
    "make", "know", "people", "think", "say", "time", "year", "way",
    "even", "may", "first", "two", "many", "much", "well", "back",
    "going", "want", "really", "still", "us", "last", "take", "come",
}

ALL_STOP_WORDS = STOP_WORDS | EXTRA_STOP_WORDS


# ---------------------------------------------------------------------------
# Text Preprocessing (mirrors the training pipeline)
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Clean and tokenise text using the same steps applied during training.

    Steps
    -----
    1. Lower-case the entire string.
    2. Remove URLs.
    3. Strip all characters except lower-case letters and whitespace.
    4. Remove English stop-words and very short tokens (<=2 chars).

    Parameters
    ----------
    text : str
        Raw input text (headline, article body, or research excerpt).

    Returns
    -------
    str
        Cleaned text ready for TF-IDF vectorisation.
    """
    if not text or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    words = [
        word for word in text.split()
        if word not in STOP_WORDS and len(word) > 2
    ]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Keyword Extraction
# ---------------------------------------------------------------------------

def extract_keywords(
    text: str,
    tfidf_vectorizer,
    top_n: int = 10,
    exclude_extra_stops: bool = True,
) -> list[dict]:
    """Extract the most important keywords from *text* using TF-IDF weights.

    The function transforms *text* with the project's fitted TF-IDF vectorizer
    and returns the features with the highest weights.

    Parameters
    ----------
    text : str
        **Pre-processed** text (output of :func:`preprocess_text`).
    tfidf_vectorizer : TfidfVectorizer
        The fitted vectorizer loaded from ``news_category_tfidf.pkl``.
    top_n : int, optional
        Maximum number of keywords to return (default 10).
    exclude_extra_stops : bool, optional
        If *True*, remove high-frequency but low-value words from the results.

    Returns
    -------
    list[dict]
        Each dict has ``{"keyword": str, "score": float}`` sorted descending
        by score.
    """
    if not text or not text.strip():
        return []

    tfidf_matrix = tfidf_vectorizer.transform([text])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Convert sparse row to dense array
    scores = tfidf_matrix.toarray().flatten()

    # Pair each feature with its score, keep only non-zero
    scored = [
        (feature_names[i], scores[i])
        for i in scores.nonzero()[0]
    ]

    if exclude_extra_stops:
        scored = [
            (word, score) for word, score in scored
            if word not in EXTRA_STOP_WORDS
            and not all(len(token) <= 2 for token in word.split())
        ]

    # Sort descending by score, take top-n
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"keyword": kw, "score": round(sc, 4)} for kw, sc in scored[:top_n]]


# ---------------------------------------------------------------------------
# Document Text Extraction (PDF and TXT)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 50) -> str:
    """Extract selectable text from a PDF file using PyMuPDF.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the uploaded PDF file.
    max_pages : int, optional
        Safety limit — only the first *max_pages* pages are processed.

    Returns
    -------
    str
        Concatenated plain text extracted from the PDF.

    Raises
    ------
    ImportError
        If PyMuPDF (``fitz``) is not installed.
    ValueError
        If the PDF contains no extractable text (likely a scanned document).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF text extraction. "
            "Install it with:  pip install PyMuPDF"
        )

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_to_read = min(len(doc), max_pages)

    text_parts = []
    for page_num in range(pages_to_read):
        page_text = doc[page_num].get_text()
        if page_text:
            text_parts.append(page_text)

    doc.close()

    full_text = "\n".join(text_parts).strip()
    if not full_text:
        raise ValueError(
            "No selectable text found in the PDF. "
            "This may be a scanned document — OCR is required to extract "
            "text from scanned images, which is not supported in this version."
        )

    return full_text


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode and return text from a plain-text (.txt) upload.

    Tries UTF-8 first, falls back to latin-1.
    """
    try:
        return file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1").strip()


def truncate_text(text: str, max_chars: int = 50_000) -> tuple[str, bool]:
    """Truncate text to *max_chars* for safe processing.

    Returns
    -------
    tuple[str, bool]
        The (possibly truncated) text and a flag indicating whether
        truncation occurred.
    """
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True
