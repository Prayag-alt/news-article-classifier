"""
app.py — News Intelligence System
A professional, academic-grade Streamlit application for multi-category
news classification, uncertainty estimation, keyword extraction, sentiment
analysis, and research-to-news recommendation.
"""

import warnings
import streamlit as st
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="News Intelligence System",
    page_icon=None,
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
# Professional CSS Design System (Navy / Blue / White Palette, No Emojis)
# ---------------------------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* ---- Sidebar Styling ---- */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown label,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 500;
        font-size: 0.92rem;
    }

    /* ---- Main Hero Card ---- */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 36px 32px;
        color: #ffffff;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #cbd5e1;
        line-height: 1.6;
        margin: 0 0 16px 0;
        max-width: 900px;
    }
    .hero-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }
    .hero-badge {
        display: inline-block;
        background-color: rgba(37, 99, 235, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.4);
        color: #93c5fd;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 6px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* ---- Metric Cards ---- */
    .metric-card-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        text-align: left;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .metric-value-primary {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-subtext {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* ---- Feature Cards ---- */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 22px 20px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        transition: border-color 0.15s ease;
    }
    .feature-card:hover {
        border-color: #93c5fd;
    }
    .feature-number {
        font-size: 0.75rem;
        font-weight: 700;
        color: #2563eb;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .feature-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0 0 8px 0;
    }
    .feature-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.5;
        margin: 0;
    }

    /* ---- Article Cards ---- */
    .article-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .article-headline {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0 0 8px 0;
        line-height: 1.4;
    }
    .article-desc {
        font-size: 0.9rem;
        color: #334155;
        line-height: 1.55;
        margin: 6px 0 10px 0;
    }
    .article-footer {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 16px;
        font-size: 0.8rem;
        color: #64748b;
        border-top: 1px solid #f1f5f9;
        padding-top: 10px;
        margin-top: 8px;
    }

    /* ---- Category Badges ---- */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-primary   { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
    .badge-secondary { background: #f8fafc; color: #334155; border: 1px solid #cbd5e1; }
    .badge-dark      { background: #0f172a; color: #f8fafc; border: 1px solid #1e293b; }

    /* ---- Confidence Badges ---- */
    .confidence-tag-high     { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .confidence-tag-moderate { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .confidence-tag-low      { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }

    /* ---- Sentiment Tags ---- */
    .sentiment-tag-positive { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; padding: 4px 12px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .sentiment-tag-negative { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; padding: 4px 12px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .sentiment-tag-neutral  { background: #f8fafc; color: #334155; border: 1px solid #cbd5e1; padding: 4px 12px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }

    /* ---- Keyword Pill ---- */
    .keyword-pill {
        display: inline-block;
        background: #f1f5f9;
        color: #1e293b;
        border: 1px solid #cbd5e1;
        padding: 4px 11px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 2px 4px 2px 0;
    }
    .keyword-pill-weight {
        color: #2563eb;
        font-weight: 600;
        margin-left: 4px;
        font-size: 0.76rem;
    }

    /* ---- Progress Bar for Similarity ---- */
    .sim-bar-bg {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        overflow: hidden;
        margin-top: 6px;
    }
    .sim-bar-fill {
        background-color: #2563eb;
        height: 100%;
        border-radius: 3px;
    }

    /* ---- Status Indicator in Sidebar ---- */
    .status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .status-dot-active {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
    }
    .status-dot-inactive {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #ef4444;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Resource Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading classification model...")
def cached_load_model():
    return load_model()


@st.cache_resource(show_spinner="Loading TF-IDF vectorizer...")
def cached_load_tfidf():
    return load_tfidf()


@st.cache_resource(show_spinner="Loading news dataset (209,527 articles)...")
def cached_load_dataset():
    return load_dataset()


@st.cache_resource(show_spinner="Computing corpus TF-IDF representation...")
def cached_corpus_vectors(_df, _tfidf):
    return build_corpus_vectors(_df, _tfidf)


# ---------------------------------------------------------------------------
# Initialize Resources
# ---------------------------------------------------------------------------
model, model_error = cached_load_model()
tfidf, tfidf_error = cached_load_tfidf()
dataset, dataset_error = cached_load_dataset()

corpus_vectors = None
if dataset is not None and tfidf is not None:
    corpus_vectors = cached_corpus_vectors(dataset, tfidf)

# ---------------------------------------------------------------------------
# Sidebar Navigation & System Status
# ---------------------------------------------------------------------------
inject_custom_css()

with st.sidebar:
    st.markdown("### News Intelligence System")
    st.caption("Machine Learning & Natural Language Processing Suite")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Overview & Metrics",
            "News Article Classifier",
            "Personalized News Feed",
            "Research-to-News Retrieval",
            "System Documentation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### System Components")

    if model is not None:
        st.markdown(
            '<div class="status-row"><span class="status-dot-active"></span> Linear SVM Model: Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-row"><span class="status-dot-inactive"></span> Linear SVM Model: Offline</div>',
            unsafe_allow_html=True,
        )

    if tfidf is not None:
        st.markdown(
            '<div class="status-row"><span class="status-dot-active"></span> TF-IDF Vectorizer: Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-row"><span class="status-dot-inactive"></span> TF-IDF Vectorizer: Offline</div>',
            unsafe_allow_html=True,
        )

    if dataset is not None:
        st.markdown(
            f'<div class="status-row"><span class="status-dot-active"></span> Corpus: {len(dataset):,} Articles</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-row"><span class="status-dot-inactive"></span> Corpus: Not Loaded</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("Linear SVM · 42 Categories · 80,000 Features")
    st.caption("Dataset Coverage: 2012–2022 (Historical)")


# ===========================================================================
# PAGE 1: OVERVIEW & BENCHMARK METRICS
# ===========================================================================
if page == "Overview & Metrics":

    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">News Intelligence System</div>
        <div class="hero-subtitle">
            An end-to-end Natural Language Processing system for fine-grained multi-category news classification,
            transparent margin-based uncertainty estimation, keyword extraction, sentiment analysis, and
            research-to-news recommendation.
        </div>
        <div class="hero-badge-row">
            <span class="hero-badge">42 News Categories</span>
            <span class="hero-badge">80,000 TF-IDF Features</span>
            <span class="hero-badge">Linear Support Vector Machine</span>
            <span class="hero-badge">209,527 Corpus Articles</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Core Capabilities")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">Capability 01</div>
            <div class="feature-title">Classification</div>
            <div class="feature-desc">Predicts one of 42 fine-grained news categories using a trained Linear SVM model.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">Capability 02</div>
            <div class="feature-title">Uncertainty Scoring</div>
            <div class="feature-desc">Calculates honest prediction confidence using raw SVM hyperplane decision score margins.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">Capability 03</div>
            <div class="feature-title">Keyword Extraction</div>
            <div class="feature-desc">Isolates salient lexical features ranked by fitted TF-IDF unigram and bigram weights.</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">Capability 04</div>
            <div class="feature-title">Sentiment Profiling</div>
            <div class="feature-desc">Computes emotional polarity (-1 to +1) and subjectivity indices on article texts.</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">Capability 05</div>
            <div class="feature-title">Smart Retrieval</div>
            <div class="feature-desc">Generates interest feeds and matches research papers against 209k articles via cosine similarity.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Empirical Benchmark Results")
    st.caption("Verified experimental performance metrics measured on the multi-class test partition.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card-container">
            <div class="metric-label">Selected Model Accuracy</div>
            <div class="metric-value-primary">61.29%</div>
            <div class="metric-subtext">Linear SVM Classifier</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card-container">
            <div class="metric-label">Macro F1-Score</div>
            <div class="metric-value-primary">0.4824</div>
            <div class="metric-subtext">Unweighted average over 42 classes</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card-container">
            <div class="metric-label">Weighted F1-Score</div>
            <div class="metric-value-primary">0.5976</div>
            <div class="metric-subtext">Support-weighted multi-class F1</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card-container">
            <div class="metric-label">Corpus Size</div>
            <div class="metric-value-primary">209,527</div>
            <div class="metric-subtext">HuffPost Articles (2012–2022)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Model Architecture Comparison")

    comparison_df = pd.DataFrame({
        "Model Architecture": [
            "Multinomial Naive Bayes",
            "Logistic Regression (L2)",
            "Linear SVM (Selected Configuration)",
        ],
        "Classification Accuracy": ["44.53%", "60.83%", "61.29%"],
        "Macro F1-Score": ["0.1653", "0.4497", "0.4824"],
        "Evaluation Status": ["Baseline", "Strong Baseline", "Production Selected"],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.info(
        "Historical Dataset Disclosure: The news dataset comprises 209,527 articles published between January 2012 and September 2022. "
        "The system operates on historical data and does not connect to live or real-time news APIs. "
        "The Linear SVM achieves 61.29% accuracy across 42 fine-grained classes, where substantial semantic overlap exists between categories "
        "(e.g., ARTS vs. CULTURE & ARTS; WORLDPOST vs. THE WORLDPOST)."
    )


# ===========================================================================
# PAGE 2: NEWS ARTICLE CLASSIFIER
# ===========================================================================
elif page == "News Article Classifier":

    st.markdown("### News Article Classification & Analysis")
    st.markdown(
        "Enter a news headline, lead paragraph, or article excerpt to categorize it across 42 classes, "
        "evaluate decision score uncertainty, extract salient keywords, and assess sentiment tone."
    )

    if model is None:
        st.error(f"Classification Model Error: {model_error}")
        st.stop()
    if tfidf is None:
        st.error(f"Vectorizer Error: {tfidf_error}")
        st.stop()

    def set_sample_text(sample: str):
        st.session_state["news_input_text"] = sample

    st.markdown("#### Pre-configured Test Samples")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.button(
            "Technology & AI",
            use_container_width=True,
            on_click=set_sample_text,
            args=(
                "Google and Microsoft announced next-generation transformer models and accelerated "
                "neural processing units designed for autonomous systems and data centers.",
            ),
        )
    with b2:
        st.button(
            "Politics & Policy",
            use_container_width=True,
            on_click=set_sample_text,
            args=(
                "The Senate committee held hearings on federal infrastructure spending, "
                "debating bipartisan tax measures and oversight on interstate highway grants.",
            ),
        )
    with b3:
        st.button(
            "Sports & League",
            use_container_width=True,
            on_click=set_sample_text,
            args=(
                "The championship finals concluded in an overtime shootout as the home franchise "
                "secured the national trophy before a sellout crowd of eighty thousand spectators.",
            ),
        )
    with b4:
        st.button(
            "Climate & Earth Science",
            use_container_width=True,
            on_click=set_sample_text,
            args=(
                "Satellite radiometer measurements published by researchers document accelerated glacial "
                "calving and anomalous thermal anomalies across Antarctic ice shelves.",
            ),
        )

    user_text = st.text_area(
        "Article Text Input",
        placeholder="Enter headline, paragraph, or full excerpt here...",
        height=140,
        key="news_input_text",
    )

    if st.button("Analyze Article", type="primary", use_container_width=True):
        if not user_text or not user_text.strip():
            st.warning("Input validation notice: Please provide article text before analysis.")
        else:
            with st.spinner("Processing text and executing inference..."):
                classification_output = classify_text(user_text, model, tfidf)

            if classification_output["error"]:
                st.error(f"Processing Error: {classification_output['error']}")
            else:
                st.markdown("---")
                st.markdown("#### Primary Classification Result")

                pred_category = classification_output["predicted_category"]
                conf_level = classification_output["confidence_level"]
                top_score = classification_output["top_score"]
                margin = classification_output["margin"]
                is_uncertain = classification_output["is_uncertain"]

                c_res1, c_res2, c_res3, c_res4 = st.columns([2, 1.2, 1.2, 1.2])
                with c_res1:
                    st.markdown(f'<span class="badge badge-dark" style="font-size:1.05rem; padding:6px 14px;">{pred_category}</span>', unsafe_allow_html=True)
                    st.caption("Predicted News Category (Out of 42)")

                with c_res2:
                    if conf_level == "High":
                        tag_html = f'<span class="confidence-tag-high">{conf_level} Confidence</span>'
                    elif conf_level == "Moderate":
                        tag_html = f'<span class="confidence-tag-moderate">{conf_level} Confidence</span>'
                    else:
                        tag_html = f'<span class="confidence-tag-low">{conf_level} Confidence</span>'
                    st.markdown(tag_html, unsafe_allow_html=True)
                    st.caption("Margin Heuristic Tier")

                with c_res3:
                    st.metric("Top Decision Score", f"{top_score:.4f}")

                with c_res4:
                    st.metric("Score Margin (1st - 2nd)", f"{margin:.4f}")

                if is_uncertain:
                    st.warning(
                        "Ambiguous Prediction Notice: The decision margin between the top two categories is small (< 0.15). "
                        "The model exhibits uncertainty across closely related candidate categories."
                    )

                # Top-5 Decision Scores
                with st.expander("Detailed Decision Score Breakdown (Top 5 Classes)"):
                    st.caption(
                        "Raw SVM decision scores represent geometric signed distances from the separating hyperplane: f(x) = w^T x + b. "
                        "They are not calibrated posterior probabilities, providing an honest unmanipulated distance metric."
                    )
                    scores_table = pd.DataFrame(classification_output["decision_scores"])
                    scores_table.columns = ["Category Name", "Decision Margin Score"]
                    scores_table.index = range(1, len(scores_table) + 1)
                    st.dataframe(scores_table, use_container_width=True)

                st.markdown("---")
                kw_col, sent_col = st.columns([1.2, 1])

                with kw_col:
                    st.markdown("#### Salient Extracted Keywords")
                    extracted_kws = extract_keywords(classification_output["cleaned_text"], tfidf, top_n=10)
                    if extracted_kws:
                        kw_html = " ".join([
                            f'<span class="keyword-pill">{kw["keyword"]}<span class="keyword-pill-weight">({kw["score"]:.3f})</span></span>'
                            for kw in extracted_kws
                        ])
                        st.markdown(kw_html, unsafe_allow_html=True)
                        st.caption("Features extracted by fitted TF-IDF unigram and bigram importance.")
                    else:
                        st.info("No discriminating vocabulary terms isolated.")

                with sent_col:
                    st.markdown("#### Sentiment Tone & Polarity")
                    sent_data = analyze_sentiment(user_text)
                    sent_label = sent_data["label"]

                    if sent_label == "Positive":
                        sent_badge = f'<span class="sentiment-tag-positive">Positive Polarity ({sent_data["polarity"]:+.3f})</span>'
                    elif sent_label == "Negative":
                        sent_badge = f'<span class="sentiment-tag-negative">Negative Polarity ({sent_data["polarity"]:+.3f})</span>'
                    else:
                        sent_badge = f'<span class="sentiment-tag-neutral">Neutral Polarity ({sent_data["polarity"]:+.3f})</span>'

                    st.markdown(sent_badge, unsafe_allow_html=True)
                    st.caption(f"Subjectivity Metric: {sent_data['subjectivity']:.3f} (Scale: 0.00 = Objective, 1.00 = Subjective)")

                with st.expander("Normalized Input View"):
                    st.code(classification_output["cleaned_text"], language=None)


# ===========================================================================
# PAGE 3: PERSONALIZED NEWS FEED
# ===========================================================================
elif page == "Personalized News Feed":

    st.markdown("### Personalized News Feed")
    st.markdown(
        "Configure your reading profile by selecting topics of interest. "
        "The system dynamically samples and aggregates relevant articles from the 209,527 historical corpus."
    )

    if dataset is None:
        st.error(f"Corpus Error: {dataset_error}")
        st.stop()

    all_categories = sorted(dataset["category"].unique().tolist())

    def set_feed_categories(cats):
        st.session_state["feed_selected_categories"] = cats

    if "feed_selected_categories" not in st.session_state:
        st.session_state["feed_selected_categories"] = ["TECH", "SCIENCE"]

    st.markdown("#### Category Interest Presets")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.button("World & Governance", use_container_width=True, on_click=set_feed_categories, args=(["POLITICS", "WORLD NEWS", "U.S. NEWS", "THE WORLDPOST"],))
    with p2:
        st.button("Science & Technology", use_container_width=True, on_click=set_feed_categories, args=(["TECH", "SCIENCE", "ENVIRONMENT", "GREEN"],))
    with p3:
        st.button("Sports & Culture", use_container_width=True, on_click=set_feed_categories, args=(["SPORTS", "ENTERTAINMENT", "COMEDY", "ARTS & CULTURE"],))
    with p4:
        st.button("Business & Finance", use_container_width=True, on_click=set_feed_categories, args=(["BUSINESS", "MONEY"],))

    selected_categories = st.multiselect(
        "Active Interest Categories",
        options=all_categories,
        default=st.session_state.get("feed_selected_categories", ["TECH", "SCIENCE"]),
        help="Select one or more topics to filter the article stream.",
    )

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        n_articles = st.slider("Maximum Articles to Retrieve", min_value=5, max_value=50, value=20, step=5)
    with ctrl2:
        sort_by_date = st.checkbox("Order by Publication Date (Most Recent First)", value=True)

    if st.button("Generate Personalized Feed", type="primary", use_container_width=True):
        if not selected_categories:
            st.warning("Please select at least one category to populate your feed.")
        else:
            with st.spinner("Retrieving articles from corpus..."):
                feed_articles = get_interest_based_feed(
                    dataset, selected_categories, n_articles=n_articles, sort_by_date=sort_by_date
                )

            if feed_articles.empty:
                st.info("No matching records found for the chosen categories.")
            else:
                st.success(f"Retrieved {len(feed_articles)} articles across {len(selected_categories)} interest categories.")

                for _, item in feed_articles.iterrows():
                    headline = item.get("headline", "Untitled")
                    desc = item.get("short_description", "")
                    cat = item.get("category", "")
                    date_val = str(item.get("date", ""))[:10] if pd.notna(item.get("date")) else "Undated"
                    author_val = item.get("authors", "")

                    st.markdown(f"""
                    <div class="article-item">
                        <div class="article-headline">{headline}</div>
                        <span class="badge badge-primary">{cat}</span>
                        <div class="article-desc">{desc}</div>
                        <div class="article-footer">
                            <span>Date: {date_val}</span>
                            {f"<span>Author: {author_val}</span>" if author_val else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 4: RESEARCH-TO-NEWS RETRIEVAL
# ===========================================================================
elif page == "Research-to-News Retrieval":

    st.markdown("### Research-to-News Semantic Retrieval")
    st.markdown(
        "Cross-reference scholarly research papers and academic abstracts with mainstream news coverage. "
        "Upload a PDF or TXT paper, or paste an abstract below to calculate TF-IDF cosine similarities against 209,527 historical articles."
    )

    if dataset is None:
        st.error(f"Corpus Error: {dataset_error}")
        st.stop()
    if tfidf is None:
        st.error(f"Vectorizer Error: {tfidf_error}")
        st.stop()
    if corpus_vectors is None:
        st.error("Corpus representation vectors are unavailable.")
        st.stop()

    st.info(
        "Historical Dataset Disclosure: Matches are computed strictly against the HuffPost 2012–2022 corpus. "
        "Results reflect historical contextual coverage and do not represent live news feeds."
    )

    tab_paste, tab_upload = st.tabs(["Paste Research Text / Abstract", "Upload Research File (PDF / TXT)"])
    research_input = ""

    with tab_paste:
        def set_research_preset(preset_text: str):
            st.session_state["research_text_input"] = preset_text

        st.markdown("#### Academic Sample Abstracts")
        sa1, sa2, sa3 = st.columns(3)
        with sa1:
            st.button(
                "Machine Learning & AI",
                use_container_width=True,
                on_click=set_research_preset,
                args=(
                    "This study explores self-supervised deep neural networks, transformer architectures, "
                    "and large language model alignment for automated natural language understanding and algorithmic fairness.",
                ),
            )
        with sa2:
            st.button(
                "Renewable Energy & Climate",
                use_container_width=True,
                on_click=set_research_preset,
                args=(
                    "We investigate photovoltaic solar cell efficiency, battery storage grid integration, "
                    "and decarbonization pathways to mitigate global greenhouse gas emissions and environmental degradation.",
                ),
            )
        with sa3:
            st.button(
                "Biomedicine & Genomics",
                use_container_width=True,
                on_click=set_research_preset,
                args=(
                    "Recent clinical trials evaluate CRISPR-Cas9 targeted genome editing therapies for "
                    "treating hereditary immunodeficiencies and chronic cellular disorders in patient cohorts.",
                ),
            )

        pasted_text = st.text_area(
            "Abstract or Research Excerpt",
            placeholder="Paste research abstract, introduction, or literature review section...",
            height=160,
            key="research_text_input",
        )
        if pasted_text and pasted_text.strip():
            research_input = pasted_text.strip()

    with tab_upload:
        uploaded_doc = st.file_uploader(
            "Select Research Document (PDF or TXT format, maximum 10 MB)",
            type=["pdf", "txt"],
            help="Extracts text via PyMuPDF for PDF documents and standard encoding for plain text.",
        )
        if uploaded_doc is not None:
            if uploaded_doc.size > 10 * 1024 * 1024:
                st.error("Document exceeds maximum size threshold of 10 MB.")
            else:
                try:
                    file_payload = uploaded_doc.read()
                    if uploaded_doc.name.lower().endswith(".pdf"):
                        extracted_doc_text = extract_text_from_pdf(file_payload, max_pages=50)
                    else:
                        extracted_doc_text = extract_text_from_txt(file_payload)

                    if extracted_doc_text.strip():
                        research_input = extracted_doc_text
                        st.success(f"Successfully extracted {len(extracted_doc_text):,} characters from '{uploaded_doc.name}'.")
                    else:
                        st.warning("No readable text discovered in uploaded document.")
                except Exception as exc:
                    st.error(f"Document extraction failure: {exc}")

    if research_input:
        st.markdown("---")
        st.markdown("#### Document Text Review")

        truncated_doc, was_truncated = truncate_text(research_input, max_chars=50000)
        if was_truncated:
            st.warning(f"Text truncated to 50,000 characters for processing stability (Original: {len(research_input):,} chars).")

        reviewed_text = st.text_area(
            "Review or refine query text prior to similarity calculation",
            value=truncated_doc,
            height=150,
            key="reviewed_research_text",
        )

        top_k = st.slider("Number of Relevant Articles to Retrieve", min_value=5, max_value=25, value=10, step=5)

        if st.button("Execute Similarity Search", type="primary", use_container_width=True):
            if not reviewed_text.strip():
                st.warning("Please provide valid research text.")
            else:
                with st.spinner("Computing TF-IDF cosine similarity against 209,527 articles..."):
                    recommendations, r_keywords = recommend_from_research(
                        reviewed_text,
                        dataset,
                        corpus_vectors,
                        tfidf,
                        top_n=top_k,
                    )

                if not recommendations:
                    st.info("No matching articles met the similarity threshold.")
                else:
                    st.markdown("#### Extracted Research Keywords")
                    if r_keywords:
                        kw_markup = " ".join([
                            f'<span class="keyword-pill">{kw["keyword"]} <span class="keyword-pill-weight">({kw["score"]:.3f})</span></span>'
                            for kw in r_keywords
                        ])
                        st.markdown(kw_markup, unsafe_allow_html=True)

                    st.markdown(f"#### Top {len(recommendations)} Relevant Historical Articles")
                    st.caption("Ranked by cosine similarity in the 80,000-dimensional TF-IDF unigram/bigram feature space.")

                    for rank, rec in enumerate(recommendations, 1):
                        sim_val = rec["similarity_score"]
                        sim_pct = min(sim_val * 100 * 2.5, 100.0)

                        st.markdown(f"""
                        <div class="article-item">
                            <div class="article-headline">{rank}. {rec['headline']}</div>
                            <span class="badge badge-primary">{rec['category']}</span>
                            <div class="article-desc">{rec['short_description']}</div>
                            <div style="margin-top: 10px;">
                                <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#475569;">
                                    <span>Cosine Similarity: <strong>{sim_val:.4f}</strong></span>
                                    <span>{rec['match_reason']}</span>
                                </div>
                                <div class="sim-bar-bg">
                                    <div class="sim-bar-fill" style="width: {sim_pct:.1f}%;"></div>
                                </div>
                            </div>
                            <div class="article-footer">
                                <span>Date: {rec['date']}</span>
                                {f"<span>Author: {rec['authors']}</span>" if rec.get('authors') else ""}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 5: SYSTEM DOCUMENTATION & ARCHITECTURE
# ===========================================================================
elif page == "System Documentation":

    st.markdown("### System Architecture & Technical Specifications")

    st.markdown("""
    #### 1. Corpus Dataset Details
    - **Source:** HuffPost News Category Dataset v3 (Curated by Rishabh Misra)
    - **Total Records:** 209,527 articles
    - **Temporal Span:** January 28, 2012 to September 23, 2022
    - **Target Categories:** 42 distinct news categories
    - **Corpus Imbalance Ratio:** ~35.11:1 (POLITICS: 35,602 vs. EDUCATION: 1,014)

    #### 2. Text Preprocessing Pipeline
    The text normalization pipeline applies deterministic cleaning steps:
    1. Case normalization to lower-case characters.
    2. Regular expression removal of web URLs (http/https/www).
    3. Removal of non-alphabetic characters and punctuation.
    4. Tokenization and filtering against NLTK English stop-words.
    5. Discarding short tokens (character length <= 2).

    #### 3. Vectorization Space
    - **Feature Representation:** TF-IDF (Term Frequency - Inverse Document Frequency)
    - **N-gram Range:** Unigrams and Bigrams (1, 2)
    - **Maximum Feature Vocabulary:** 80,000 dimensions
    - **Matrix Structure:** Sparse CSR representation

    #### 4. Classification Model Benchmarks
    | Model Architecture | Accuracy | Macro F1-Score | Weighted F1-Score |
    |---|---:|---:|---:|
    | Multinomial Naive Bayes | 44.53% | 0.1653 | 0.4012 |
    | Logistic Regression (L2) | 60.83% | 0.4497 | 0.5891 |
    | **Linear SVM (Production)** | **61.29%** | **0.4824** | **0.5976** |

    #### 5. Uncertainty & Decision Margin Rationale
    Support Vector Machines maximize the geometric margin between classes. In multi-class One-vs-Rest (OvR) mode,
    the model yields signed decision scores:
    
    $$f_k(x) = w_k^T x + b_k$$
    
    Rather than converting scores into synthetic probabilities using uncalibrated Softmax (which introduces artificial certainty),
    this system defines confidence strictly based on the separation margin:
    
    $$\Delta = f_{\text{top 1}}(x) - f_{\text{top 2}}(x)$$

    - **High Confidence:** $\Delta \ge 0.40$
    - **Moderate Confidence:** $0.15 \le \Delta < 0.40$
    - **Low Confidence (Ambiguous):** $\Delta < 0.15$

    #### 6. System Limitations
    - **Temporal Scope:** Articles date from 2012 to 2022; recent events (post-September 2022) are not indexed.
    - **Category Overlap:** High semantic proximity between classes like ARTS vs. ARTS & CULTURE introduces inherent ambiguity.
    - **Non-OCR PDF Support:** Document parsing requires digital selectable text; image-only scanned PDFs require external OCR.
    """)

    st.markdown("---")
    st.caption("News Intelligence System · Academic NLP Implementation · Zero Fabricated Metrics Guarantee")
