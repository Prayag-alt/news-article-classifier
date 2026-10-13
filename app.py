"""
app.py — News Intelligence System
===================================
A multi-page Streamlit application for news classification, sentiment
analysis, keyword extraction, and personalised news recommendation.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page Configuration (MUST be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="News Intelligence System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports from project modules
# ---------------------------------------------------------------------------
from news_classifier import (
    load_model,
    load_tfidf,
    classify_text,
    analyze_sentiment,
)
from text_processing import (
    preprocess_text,
    extract_keywords,
    extract_text_from_pdf,
    extract_text_from_txt,
    truncate_text,
)
from recommendation import (
    load_dataset,
    get_interest_based_feed,
    build_corpus_vectors,
    recommend_from_research,
)


# ---------------------------------------------------------------------------
# Custom CSS for a polished news-oriented design
# ---------------------------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #e0e0e0 !important;
    }

    /* ---- Category Badges ---- */
    .category-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        color: white;
        margin: 2px 4px 2px 0;
        letter-spacing: 0.3px;
    }
    .badge-politics      { background: #e74c3c; }
    .badge-business      { background: #2ecc71; }
    .badge-tech          { background: #3498db; }
    .badge-science       { background: #9b59b6; }
    .badge-sports        { background: #e67e22; }
    .badge-entertainment { background: #e91e63; }
    .badge-health        { background: #00bcd4; }
    .badge-travel        { background: #ff9800; }
    .badge-food          { background: #8bc34a; }
    .badge-default       { background: #607d8b; }

    /* ---- Metric Cards ---- */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .metric-card h3 { margin: 0 0 8px 0; font-size: 0.9rem; opacity: 0.9; }
    .metric-card .metric-value { font-size: 2rem; font-weight: 700; }

    /* ---- Article Card ---- */
    .article-card {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        transition: box-shadow 0.2s ease;
    }
    .article-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .article-card h4 {
        margin: 0 0 8px 0;
        color: #1a1a2e;
        font-size: 1.05rem;
    }
    .article-card p {
        color: #555;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 4px 0;
    }
    .article-meta {
        font-size: 0.78rem;
        color: #888;
        margin-top: 6px;
    }

    /* ---- Confidence Indicator ---- */
    .confidence-high     { color: #2ecc71; font-weight: 700; }
    .confidence-moderate { color: #f39c12; font-weight: 700; }
    .confidence-low      { color: #e74c3c; font-weight: 700; }

    /* ---- Keyword Tag ---- */
    .keyword-tag {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px 4px;
        border: 1px solid #c7d2fe;
    }

    /* ---- Sentiment Pill ---- */
    .sentiment-positive { background: #d4edda; color: #155724; padding: 6px 16px; border-radius: 20px; font-weight: 600; display: inline-block; }
    .sentiment-negative { background: #f8d7da; color: #721c24; padding: 6px 16px; border-radius: 20px; font-weight: 600; display: inline-block; }
    .sentiment-neutral  { background: #e2e3e5; color: #383d41; padding: 6px 16px; border-radius: 20px; font-weight: 600; display: inline-block; }

    /* ---- Hero Section ---- */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 20px;
        padding: 48px 40px;
        color: white;
        margin-bottom: 32px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    .hero h1 { font-size: 2.2rem; margin-bottom: 12px; font-weight: 700; }
    .hero p  { font-size: 1.08rem; opacity: 0.85; line-height: 1.7; }

    /* ---- Feature Card ---- */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 14px;
        padding: 28px 24px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }
    .feature-icon { font-size: 2.4rem; margin-bottom: 12px; }
    .feature-card h4 { color: #1a1a2e; margin: 8px 0; font-size: 1rem; }
    .feature-card p  { color: #666; font-size: 0.85rem; line-height: 1.5; }

    /* ---- Similarity Bar ---- */
    .sim-bar-bg {
        background: #e8e8e8;
        border-radius: 6px;
        height: 8px;
        width: 100%;
        margin-top: 6px;
    }
    .sim-bar-fill {
        height: 8px;
        border-radius: 6px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Category badge colour
# ---------------------------------------------------------------------------
CATEGORY_BADGE_MAP = {
    "POLITICS": "badge-politics",
    "BUSINESS": "badge-business",
    "MONEY": "badge-business",
    "TECH": "badge-tech",
    "SCIENCE": "badge-science",
    "SPORTS": "badge-sports",
    "ENTERTAINMENT": "badge-entertainment",
    "COMEDY": "badge-entertainment",
    "HEALTHY LIVING": "badge-health",
    "WELLNESS": "badge-health",
    "TRAVEL": "badge-travel",
    "FOOD & DRINK": "badge-food",
    "TASTE": "badge-food",
}


def category_badge(cat: str) -> str:
    """Return an HTML badge span for a category name."""
    css_class = CATEGORY_BADGE_MAP.get(cat, "badge-default")
    return f'<span class="category-badge {css_class}">{cat}</span>'


# ---------------------------------------------------------------------------
# Cached Resource Loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading classification model…")
def cached_load_model():
    return load_model()


@st.cache_resource(show_spinner="Loading TF-IDF vectorizer…")
def cached_load_tfidf():
    return load_tfidf()


@st.cache_resource(show_spinner="Loading news dataset (~209k articles)…")
def cached_load_dataset():
    return load_dataset()


@st.cache_resource(show_spinner="Pre-computing article vectors for recommendations…")
def cached_corpus_vectors(_df, _tfidf):
    """Build TF-IDF vectors for the whole corpus (cached once per session)."""
    return build_corpus_vectors(_df, _tfidf)


# ---------------------------------------------------------------------------
# Initialise Resources
# ---------------------------------------------------------------------------

model, model_error = cached_load_model()
tfidf, tfidf_error = cached_load_tfidf()
dataset, dataset_error = cached_load_dataset()

# Pre-compute corpus vectors if both dataset and vectorizer are available
corpus_vectors = None
if dataset is not None and tfidf is not None:
    corpus_vectors = cached_corpus_vectors(dataset, tfidf)


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
inject_custom_css()

with st.sidebar:
    st.markdown("## 📰 News Intelligence")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Home Dashboard",
            "🔍 News Classifier",
            "📋 Personalized News Feed",
            "🔬 Research-to-News",
            "ℹ️ About the Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    # Status indicators
    st.markdown("### System Status")
    if model is not None:
        st.success("✅ Model loaded", icon="🤖")
    else:
        st.error("❌ Model missing")

    if tfidf is not None:
        st.success("✅ Vectorizer loaded", icon="📊")
    else:
        st.error("❌ Vectorizer missing")

    if dataset is not None:
        st.success(f"✅ Dataset: {len(dataset):,} articles", icon="📰")
    else:
        st.warning("⚠️ Dataset not loaded")

    st.markdown("---")
    st.caption("TF-IDF + Linear SVM · 42 Categories")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: HOME DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home Dashboard":

    # Hero Section
    st.markdown("""
    <div class="hero">
        <h1>📰 News Intelligence System</h1>
        <p>
            Automated news categorisation, prediction uncertainty analysis,
            keyword extraction, sentiment analysis, and personalised
            recommendation — powered by machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    st.markdown("### ✨ Core Features")

    cols = st.columns(5)
    features = [
        ("🏷️", "Category Classification",
         "Classify news into 42 categories using a trained Linear SVM model."),
        ("📊", "Prediction Confidence",
         "Inspect decision scores and margin-based uncertainty indicators."),
        ("🔑", "Keyword Extraction",
         "Surface the most important terms using TF-IDF feature weights."),
        ("💬", "Sentiment Analysis",
         "Detect positive, negative, or neutral tone in news text."),
        ("🎯", "Smart Recommendations",
         "Get personalised news by interests or by uploading research papers."),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How It Works
    st.markdown("### ⚙️ How the System Works")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Classification Pipeline**
        1. User enters a news headline or article text.
        2. Text is cleaned: lowercased, URLs removed, non-alphabetic
           characters stripped, stop-words filtered.
        3. The cleaned text is transformed into a TF-IDF vector (unigrams
           + bigrams, 80 000 features).
        4. The Linear SVM classifier predicts one of 42 news categories.
        5. Decision-function scores reveal prediction uncertainty.
        """)

    with col2:
        st.markdown("""
        **Recommendation Pipeline**
        1. The user selects preferred categories **or** uploads a research
           paper / pastes research text.
        2. For interest-based feeds, articles are filtered by category and
           sorted by date.
        3. For research-to-news, the text is vectorised with the same TF-IDF
           model and compared to all ~209 000 articles via cosine similarity.
        4. The top-ranked articles are returned with similarity scores and
           matching-keyword explanations.
        """)

    # Baseline Metrics
    st.markdown("### 📈 Baseline Model Performance")
    st.info(
        "These are the **original experimental results** from the trained "
        "Linear SVM. The model has not been retrained or modified."
    )

    m_cols = st.columns(4)
    metrics = [
        ("Linear SVM Accuracy", "61.29%"),
        ("Macro F1-Score", "0.4824"),
        ("Weighted F1-Score", "0.5976"),
        ("Categories", "42"),
    ]
    for col, (label, value) in zip(m_cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Comparison Table
    st.markdown("### 🔬 Model Comparison (Original Experiment)")
    comparison_df = pd.DataFrame({
        "Model": ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM (selected)"],
        "Accuracy": ["44.53%", "60.83%", "61.29%"],
    })
    st.table(comparison_df)

    st.warning(
        "⚠️ **Limitations:** The classifier has moderate accuracy due to "
        "class imbalance and semantic overlap between some categories "
        "(e.g., ARTS vs. CULTURE & ARTS, WORLDPOST vs. THE WORLDPOST). "
        "It does **not** detect fake news, verify facts, or assess "
        "article credibility."
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: NEWS CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 News Classifier":

    st.markdown("# 🔍 News Article Classifier")
    st.markdown(
        "Enter a news headline, article, or description to classify it "
        "into one of **42 categories**, view prediction uncertainty, "
        "extract keywords, and analyse sentiment."
    )

    # Check that model and vectorizer are loaded
    if model is None:
        st.error(f"🚫 {model_error}")
        st.stop()
    if tfidf is None:
        st.error(f"🚫 {tfidf_error}")
        st.stop()

    # Quick Presets for easy testing
    def set_sample_article(sample: str):
        st.session_state["article_input_area"] = sample

    st.markdown("##### 💡 Or try a sample article:")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.button(
            "💻 Tech Innovation",
            use_container_width=True,
            on_click=set_sample_article,
            args=(
                "Apple announced its next-generation artificial intelligence chips and "
                "neural engine architecture for smartphones, enhancing battery performance.",
            ),
        )
    with col_s2:
        st.button(
            "🏛️ Politics & Policy",
            use_container_width=True,
            on_click=set_sample_article,
            args=(
                "The Senate passed a landmark bipartisan infrastructure bill allocating billions "
                "in federal funding for public transport, highway repairs, and clean power grids.",
            ),
        )
    with col_s3:
        st.button(
            "⚽ Sports & League",
            use_container_width=True,
            on_click=set_sample_article,
            args=(
                "The championship tournament concluded with a dramatic overtime victory as the star "
                "striker scored the game-winning goal in front of 80,000 cheering fans.",
            ),
        )
    with col_s4:
        st.button(
            "🌿 Climate & Science",
            use_container_width=True,
            on_click=set_sample_article,
            args=(
                "NASA scientists published new satellite observations revealing unprecedented glacier "
                "melting and rising ocean temperatures across polar regions.",
            ),
        )

    article_text = st.text_area(
        "📝 Enter News Text",
        placeholder=(
            "Example: The government announced new economic policies to "
            "boost growth in the technology sector amid global inflation concerns."
        ),
        height=140,
        key="article_input_area",
    )

    if st.button("🚀 Analyse", type="primary", use_container_width=True):
        if not article_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Classifying…"):
                result = classify_text(article_text, model, tfidf)

            if result["error"]:
                st.error(result["error"])
            else:
                # ---- Predicted Category ----
                st.markdown("---")
                st.markdown("### 🏷️ Predicted Category")
                st.markdown(
                    category_badge(result["predicted_category"]),
                    unsafe_allow_html=True,
                )

                # ---- Confidence / Uncertainty ----
                st.markdown("### 📊 Prediction Confidence")

                conf_class = {
                    "High": "confidence-high",
                    "Moderate": "confidence-moderate",
                    "Low": "confidence-low",
                }[result["confidence_level"]]

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(
                        f'**Confidence Level:** '
                        f'<span class="{conf_class}">{result["confidence_level"]}</span>',
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.metric("Top Decision Score", result["top_score"])
                with col_c:
                    st.metric("Score Margin (1st − 2nd)", result["margin"])

                if result["is_uncertain"]:
                    st.warning(
                        "⚠️ **Low-confidence prediction.** The margin between "
                        "the top two scores is small, meaning the model is "
                        "uncertain. Consider manual review."
                    )

                # Top-5 scores
                with st.expander("📋 Top-5 Decision Scores"):
                    st.caption(
                        "These are raw SVM decision-function scores, **not "
                        "probabilities**. A higher score means the model "
                        "favours that category more. The margin between the "
                        "top two scores is used as a heuristic confidence "
                        "indicator."
                    )
                    scores_df = pd.DataFrame(result["decision_scores"])
                    scores_df.columns = ["Category", "Decision Score"]
                    scores_df.index = range(1, len(scores_df) + 1)
                    st.dataframe(scores_df, use_container_width=True)

                # ---- Keywords ----
                st.markdown("### 🔑 Extracted Keywords")
                keywords = extract_keywords(
                    result["cleaned_text"], tfidf, top_n=10
                )
                if keywords:
                    kw_html = " ".join(
                        f'<span class="keyword-tag">{kw["keyword"]}</span>'
                        for kw in keywords
                    )
                    st.markdown(kw_html, unsafe_allow_html=True)
                else:
                    st.info("No significant keywords found for this input.")

                # ---- Sentiment ----
                st.markdown("### 💬 Sentiment Analysis")
                sentiment = analyze_sentiment(article_text)

                sent_class = f"sentiment-{sentiment['label'].lower()}"
                st.markdown(
                    f'<span class="{sent_class}">'
                    f'{sentiment["label"]}  (polarity: {sentiment["polarity"]}, '
                    f'subjectivity: {sentiment["subjectivity"]})</span>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Sentiment is analysed using TextBlob, a rule-based "
                    "approach. It estimates emotional tone — not factual "
                    "accuracy. Polarity ranges from −1 (very negative) to "
                    "+1 (very positive). Subjectivity ranges from 0 "
                    "(objective) to 1 (subjective). Results on news text "
                    "may differ from conversational text."
                )

                # ---- Preprocessed Text ----
                with st.expander("🔧 View Preprocessed Text"):
                    st.code(result["cleaned_text"], language=None)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: PERSONALIZED NEWS FEED
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📋 Personalized News Feed":

    st.markdown("# 📋 Personalised News Feed")
    st.markdown(
        "Select your preferred news categories to build a personalised "
        "feed from the dataset of ~209 000 articles."
    )

    if dataset is None:
        st.error(f"🚫 {dataset_error}")
        st.stop()

    # Available categories
    all_categories = sorted(dataset["category"].unique().tolist())

    # Popular presets
    st.markdown("#### Quick Presets")

    def set_feed_preset(cats):
        st.session_state["feed_selected_categories"] = cats

    if "feed_selected_categories" not in st.session_state:
        st.session_state["feed_selected_categories"] = ["TECH", "SCIENCE"]

    preset_cols = st.columns(4)
    presets = {
        "🌍 World & Politics": ["POLITICS", "WORLD NEWS", "WORLDPOST", "THE WORLDPOST", "U.S. NEWS"],
        "💻 Tech & Science": ["TECH", "SCIENCE", "ENVIRONMENT", "GREEN"],
        "🏈 Sports & Fun": ["SPORTS", "COMEDY", "ENTERTAINMENT", "WEIRD NEWS"],
        "💰 Business & Money": ["BUSINESS", "MONEY"],
    }
    for col, (label, cats) in zip(preset_cols, presets.items()):
        with col:
            st.button(
                label,
                use_container_width=True,
                on_click=set_feed_preset,
                args=(cats,),
            )

    # Multi-select bound to current selection
    selected_categories = st.multiselect(
        "Choose categories",
        options=all_categories,
        default=st.session_state.get("feed_selected_categories", ["TECH", "SCIENCE"]),
        help="Select one or more categories to populate your feed.",
    )

    # Controls
    col_n, col_sort = st.columns(2)
    with col_n:
        n_articles = st.slider("Number of articles", 5, 50, 20)
    with col_sort:
        sort_by_date = st.checkbox("Sort by date (newest first)", value=True)

    if st.button("🔄 Load Feed", type="primary", use_container_width=True):
        if not selected_categories:
            st.warning("Please select at least one category.")
        else:
            with st.spinner("Fetching articles…"):
                feed = get_interest_based_feed(
                    dataset, selected_categories, n_articles, sort_by_date
                )

            if feed.empty:
                st.info("No articles found for the selected categories.")
            else:
                st.success(f"Showing **{len(feed)}** articles from **{len(selected_categories)}** categories.")

                for _, row in feed.iterrows():
                    headline = row.get("headline", "Untitled")
                    desc = row.get("short_description", "")
                    cat = row.get("category", "")
                    date_str = str(row.get("date", ""))[:10] if pd.notna(row.get("date")) else ""
                    authors = row.get("authors", "")

                    st.markdown(f"""
                    <div class="article-card">
                        <h4>{headline}</h4>
                        {category_badge(cat)}
                        <p>{desc}</p>
                        <div class="article-meta">
                            {f"📅 {date_str}" if date_str and date_str != "NaT" else ""}
                            {f" · ✍️ {authors}" if authors else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: RESEARCH-TO-NEWS RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Research-to-News":

    st.markdown("# 🔬 Research-to-News Recommendations")
    st.markdown(
        "Upload a research paper (PDF/TXT) or paste an abstract/excerpt "
        "to discover related news articles from our dataset."
    )

    if dataset is None:
        st.error(f"🚫 {dataset_error}")
        st.stop()
    if tfidf is None:
        st.error(f"🚫 {tfidf_error}")
        st.stop()
    if corpus_vectors is None:
        st.error("🚫 Corpus vectors could not be computed. Check that the dataset and vectorizer are loaded.")
        st.stop()

    st.info(
        "📌 **Note:** Recommendations are retrieved from the HuffPost News "
        "Category Dataset (2012 – 2022). Results reflect historical coverage "
        "and are **not live news**."
    )

    # ---- Input Tabs ----
    tab_paste, tab_upload = st.tabs(["📝 Paste Text", "📄 Upload File"])

    research_text = ""

    with tab_paste:
        def set_research_sample(sample_text: str):
            st.session_state["research_pasted_area"] = sample_text

        st.markdown("##### 💡 Or try a sample research abstract:")
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.button(
                "🤖 AI & Deep Learning",
                use_container_width=True,
                on_click=set_research_sample,
                args=(
                    "This study investigates deep learning neural networks and transformers "
                    "for natural language processing, automated text classification, and algorithmic efficiency.",
                ),
            )
        with col_r2:
            st.button(
                "🌱 Climate & Energy",
                use_container_width=True,
                on_click=set_research_sample,
                args=(
                    "We analyze the transition towards renewable solar and wind energy grids to mitigate global "
                    "greenhouse gas emissions, climate warming, and ecological disruptions.",
                ),
            )
        with col_r3:
            st.button(
                "🧬 Medical Genomics",
                use_container_width=True,
                on_click=set_research_sample,
                args=(
                    "This paper presents clinical trials and genomic sequencing techniques utilizing CRISPR "
                    "gene editing therapies to treat hereditary cellular disorders and immune diseases.",
                ),
            )

        pasted = st.text_area(
            "Paste your research abstract, introduction, or selected paragraphs:",
            height=180,
            key="research_pasted_area",
            placeholder=(
                "Example: This paper investigates the impact of artificial "
                "intelligence on climate change mitigation strategies…"
            ),
        )
        if pasted and pasted.strip():
            research_text = pasted.strip()

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a PDF or TXT file (max 10 MB)",
            type=["pdf", "txt"],
            help="Supported formats: PDF (with selectable text) and plain text.",
        )
        if uploaded_file is not None:
            # Validate file size (10 MB)
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File too large. Please upload a file under 10 MB.")
            else:
                try:
                    file_bytes = uploaded_file.read()
                    if uploaded_file.name.lower().endswith(".pdf"):
                        extracted = extract_text_from_pdf(file_bytes, max_pages=50)
                    else:
                        extracted = extract_text_from_txt(file_bytes)

                    if extracted.strip():
                        research_text = extracted
                        st.success(
                            f"✅ Extracted **{len(extracted):,}** characters "
                            f"from `{uploaded_file.name}`."
                        )
                    else:
                        st.warning("The file appears to be empty or contains no extractable text.")

                except ImportError as ie:
                    st.error(str(ie))
                except ValueError as ve:
                    st.warning(str(ve))
                except Exception as exc:
                    st.error(f"Failed to process file: {exc}")

    # ---- Text Preview & Editing ----
    if research_text:
        st.markdown("### 📖 Extracted Text Preview")

        truncated_text, was_truncated = truncate_text(research_text, max_chars=50_000)
        if was_truncated:
            st.warning(
                f"⚠️ The text was truncated to 50 000 characters for safe "
                f"processing (original: {len(research_text):,} characters)."
            )

        edited_text = st.text_area(
            "Review and optionally edit the text before analysis:",
            value=truncated_text,
            height=200,
            key="research_edit",
        )

        # Controls
        col_top_n, _ = st.columns([1, 2])
        with col_top_n:
            top_n = st.slider("Number of recommendations", 5, 30, 15)

        if st.button("🔍 Find Related News", type="primary", use_container_width=True):
            if not edited_text.strip():
                st.warning("The text is empty. Please enter or upload research text.")
            else:
                with st.spinner("Analysing text and searching articles…"):
                    recommendations, research_keywords = recommend_from_research(
                        edited_text,
                        dataset,
                        corpus_vectors,
                        tfidf,
                        top_n=top_n,
                    )

                if not recommendations:
                    st.info(
                        "No related articles found. The research topic may "
                        "not overlap with the news dataset's coverage."
                    )
                else:
                    # ---- Research Keywords ----
                    st.markdown("### 🔑 Identified Research Keywords")
                    if research_keywords:
                        kw_html = " ".join(
                            f'<span class="keyword-tag">{kw["keyword"]}</span>'
                            for kw in research_keywords
                        )
                        st.markdown(kw_html, unsafe_allow_html=True)

                    # ---- Recommendations ----
                    st.markdown(f"### 📰 Top {len(recommendations)} Related News Articles")
                    st.caption(
                        "Articles are ranked by **cosine similarity** between "
                        "TF-IDF vectors. The similarity score is a measure of "
                        "textual overlap — not a probability or relevance guarantee."
                    )

                    for i, rec in enumerate(recommendations, 1):
                        sim_pct = rec["similarity_score"] * 100
                        st.markdown(f"""
                        <div class="article-card">
                            <h4>{i}. {rec["headline"]}</h4>
                            {category_badge(rec["category"])}
                            <p>{rec["short_description"]}</p>
                            <div class="article-meta">
                                📅 {rec["date"]}
                                {f" · ✍️ {rec['authors']}" if rec.get("authors") else ""}
                            </div>
                            <div style="margin-top: 8px;">
                                <strong>Similarity:</strong> {rec["similarity_score"]:.4f}
                                <div class="sim-bar-bg">
                                    <div class="sim-bar-fill" style="width: {min(sim_pct * 3, 100):.0f}%;"></div>
                                </div>
                            </div>
                            <div style="margin-top: 6px; font-size: 0.82rem; color: #666;">
                                💡 {rec["match_reason"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: ABOUT THE PROJECT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About the Project":

    st.markdown("# ℹ️ About the Project")

    st.markdown("""
    ## 📚 Dataset

    **Name:** News Category Dataset v3 (HuffPost)
    **Source:** [Kaggle — Rishabh Misra](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
    **Articles:** ~209 000 news articles (2012 – 2022)
    **Categories:** 42 news categories
    **Fields:** headline, short_description, category, authors, date, link

    ---

    ## 🔧 Text Preprocessing Pipeline

    The following cleaning steps are applied to every input text, mirroring
    the process used during model training:

    1. **Lowercase** the entire text.
    2. **Remove URLs** (http/https/www patterns).
    3. **Remove non-alphabetic characters** — digits, punctuation, and
       special symbols are replaced with spaces.
    4. **Tokenise** by whitespace.
    5. **Remove English stop-words** (NLTK's standard list).
    6. **Remove very short tokens** (≤ 2 characters).

    The cleaned tokens are re-joined into a single string for vectorisation.

    ---

    ## 📊 TF-IDF Vectorisation

    | Parameter | Value |
    |---|---|
    | N-gram range | Unigrams + Bigrams (1, 2) |
    | Max features | 80 000 |
    | Vectorizer type | `TfidfVectorizer` (scikit-learn) |

    TF-IDF (Term Frequency–Inverse Document Frequency) converts text into
    numerical feature vectors.  Each dimension represents a word or bigram,
    and its value reflects how important that term is to the document
    relative to the entire corpus.

    ---

    ## 🤖 Linear SVM Classifier

    The classifier is a **Linear Support Vector Machine** (`LinearSVC` from
    scikit-learn).  SVMs find the optimal hyperplane that separates classes
    in the high-dimensional TF-IDF feature space.

    For multi-class classification, scikit-learn uses a **one-vs-rest (OVR)**
    strategy — training 42 binary classifiers, one per category.

    **Why Linear SVM?**
    - Scales well to large, sparse feature matrices.
    - Competitive accuracy on text classification tasks.
    - Produces interpretable decision-function scores.

    ---

    ## 📈 Baseline Evaluation Metrics

    These results were obtained from the original training experiment and
    have **not** been modified:

    | Model | Accuracy |
    |---|---:|
    | Multinomial Naive Bayes | 44.53% |
    | Logistic Regression | 60.83% |
    | **Linear SVM (selected)** | **61.29%** |

    | Metric | Value |
    |---|---:|
    | Macro F1-Score | 0.4824 |
    | Weighted F1-Score | 0.5976 |

    The moderate accuracy is expected given:
    - **42 categories** — a fine-grained classification problem.
    - **Class imbalance** — some categories have far more articles than
      others (e.g., POLITICS dominates).
    - **Semantic overlap** — categories like *ARTS* vs *ARTS & CULTURE*,
      *WORLDPOST* vs *THE WORLDPOST*, and *PARENTS* vs *PARENTING* are
      inherently ambiguous.

    ---

    ## 💬 Sentiment Analysis

    Sentiment is analysed using **TextBlob**, a rule-based NLP library.
    It provides:
    - **Polarity** (−1 to +1): negative → positive
    - **Subjectivity** (0 to 1): objective → subjective

    **Limitations:**
    - Designed for product reviews and conversational text; may be less
      accurate on formal news writing.
    - Does not indicate factual accuracy or credibility.
    - Sarcasm and irony are generally not detected.

    ---

    ## 🔮 Limitations & Future Improvements

    **Current Limitations:**
    - The classifier was trained on historical HuffPost data and may not
      generalise well to other news sources or topics.
    - The recommendation engine retrieves from the fixed dataset (2012 –
      2022) — it does not access live news.
    - Scanned PDFs (without selectable text) require OCR, which is not
      included in this version.
    - Sentiment analysis is rule-based and may not capture nuanced tones.

    **Possible Future Work:**
    - Retrain with transformer-based models (BERT, DistilBERT) for higher
      accuracy.
    - Add calibrated probability estimates using Platt scaling or isotonic
      regression.
    - Integrate a live news API (e.g., NewsAPI, GDELT) for real-time
      recommendations.
    - Add OCR support for scanned PDFs using Tesseract.
    - Implement user accounts to persist preferences across sessions.
    - Apply SMOTE or class-weight balancing to address class imbalance.
    """)

    st.markdown("---")
    st.caption(
        "News Intelligence System · Automated News Categorisation, Analysis "
        "and Personalised Recommendation · Academic ML Project"
    )
