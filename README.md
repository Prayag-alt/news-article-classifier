# 📰 News Intelligence System: Automated News Categorization, Analysis & Personalized Recommendations

A Machine Learning and Natural Language Processing (NLP) system that automatically classifies news articles into 42 categories, evaluates prediction uncertainty using decision margins, extracts discriminative keywords, performs sentiment analysis, and delivers personalized recommendations — including a Research-to-News recommendation pipeline matching research papers with historical news coverage.

Built with **Python**, **scikit-learn**, **TF-IDF (80,000 features)**, a trained **Linear SVM**, and an interactive multi-page **Streamlit** dashboard.

---

## 📌 Project Overview

News organizations and aggregators publish hundreds of thousands of articles across diverse domains. Manually sorting, cataloguing, and cross-referencing news with research publications is labor-intensive.

This system provides an end-to-end intelligence suite:
1. **News Classification:** Predicts one of 42 fine-grained categories using a trained Linear Support Vector Machine.
2. **Prediction Confidence & Uncertainty:** Uses raw SVM decision-function scores and margin analysis (1st vs 2nd score) to detect ambiguous predictions.
3. **Keyword Extraction:** Extracts top 5–10 salient keywords based on fitted TF-IDF weights.
4. **Sentiment Analysis:** Analyzes emotional polarity and subjectivity using TextBlob.
5. **Personalized News Recommendations:**
   - **Interest-Based Feed (Mode A):** Filters and samples articles by user-selected categories with date sorting.
   - **Research-to-News (Mode B):** Ingests research abstracts or uploaded PDF/TXT files, vectorizes the text with TF-IDF, computes cosine similarities against ~209,000 articles, and displays matching articles with similarity scores and shared-term rationales.

---

## 📊 Dataset Statistics

The project uses the **HuffPost News Category Dataset v3** (Rishabh Misra, Kaggle):

| Property | Value |
|---|---:|
| Total Articles | 209,527 |
| Articles after Preprocessing | 209,034 |
| Categories | 42 |
| Date Range | 2012-01-28 to 2022-09-23 |
| Vocabulary / Features | 80,000 (Unigrams + Bigrams) |
| Dominant Category | POLITICS (35,602 articles) |
| Smallest Category | EDUCATION (1,014 articles) |
| Imbalance Ratio | ~35.11:1 |

> 📌 **Important Historical Disclosure:** Articles span 2012 to 2022. The dataset provides historical news context and does not connect to live, real-time news feeds.

---

## 📈 Baseline Model Performance & Evaluation

These are the measured evaluation metrics from the original experiments:

### Model Comparison
| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Multinomial Naive Bayes | 44.53% | 0.1653 |
| Logistic Regression | 60.83% | 0.4497 |
| **Linear SVM (Selected)** | **61.29%** | **0.4824** |

### Selected Linear SVM Metrics
| Metric | Score |
|---|---:|
| **Accuracy** | **61.29%** |
| **Macro Precision** | **52.63%** |
| **Macro Recall** | **45.92%** |
| **Macro F1-Score** | **0.4824** |
| **Weighted F1-Score** | **0.5976** |
| Hyperparameter C | 1.0 |

---

## 🚀 Quick Start & Local Execution

### 1. Prerequisites
- **Python 3.9+** (Tested on Python 3.10)
- **pip** package manager

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Prayag-alt/news-article-classifier.git
cd news-article-classifier

# Install dependencies
pip install -r requirements.txt
```

### 3. Required Model and Dataset Files
| File | Description | Location |
|---|---|---|
| `news_category_svm.pkl` | Trained Linear SVM classifier (~26 MB) | Project root |
| `news_category_tfidf.pkl` | Fitted TF-IDF vectorizer (~3 MB) | Project root |
| `News_Category_Dataset_v3.json.gz` | Compressed dataset (~26 MB) | Project root |
| `News_Category_Dataset_v3.json` | Original uncompressed dataset (~83 MB, optional) | Root or parent |

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment to Streamlit Community Cloud

The repository is configured for direct deployment on [Streamlit Community Cloud](https://share.streamlit.io):

1. **Repository:** Ensure your repository is public or linked to your Streamlit Cloud account:
   - URL: `https://github.com/Prayag-alt/news-article-classifier`
2. **Deploy on Streamlit Cloud:**
   - Log in at [share.streamlit.io](https://share.streamlit.io)
   - Click **"Create app"** / **"New app"**
   - **Repository:** `Prayag-alt/news-article-classifier`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Click **"Deploy!"**.
   Streamlit Cloud installs `requirements.txt`, loads the cached SVM model, vectorizer, and dataset, and launches the live public web application.

---

## 📁 Repository Structure

```
news-article-classifier/
├── .streamlit/
│   └── config.toml               # Streamlit theme and server configuration
├── app.py                        # Multi-page Streamlit application
├── news_classifier.py            # Model loading, classification, and sentiment analysis
├── text_processing.py            # Text preprocessing, keyword extraction, and PDF/TXT parsers
├── recommendation.py             # Interest-based feed and Research-to-News recommendation engine
├── requirements.txt              # Production Python dependencies
├── .gitignore                    # Git ignore file
├── README.md                     # Complete project documentation
├── news_category_svm.pkl         # Trained Linear SVM model file
├── news_category_tfidf.pkl       # Fitted TF-IDF vectorizer
└── News_Category_Dataset_v3.json.gz # Compressed news dataset (209k articles)
```

---

## 🧪 Testing Checklist (All 5 Features)

- [x] **Feature 1: News Classification**
  - Enter news text or click quick sample buttons (e.g. "Tech Innovation", "Politics & Policy").
  - Click **Analyse** → Model predicts category (e.g., TECH, POLITICS).
- [x] **Feature 2: Prediction Confidence & Uncertainty**
  - Decision margin between 1st and 2nd scores determines High, Moderate, or Low confidence.
  - Transparently identifies raw decision scores as distance metrics, not probabilities.
- [x] **Feature 3: Keyword Extraction**
  - Surfaces 5–10 relevant keywords with top TF-IDF feature weights.
- [x] **Feature 4: Sentiment Analysis**
  - Computes polarity (-1.0 to +1.0) and subjectivity (0.0 to 1.0) with positive/neutral/negative pills.
- [x] **Feature 5A: Personalized Feed**
  - Select categories or presets ("World & Politics", "Tech & Science", etc.).
  - Click **Load Feed** → Displays real articles with headlines, badges, dates, and authors.
- [x] **Feature 5B: Research-to-News Recommendations**
  - Paste an abstract or upload a PDF/TXT file.
  - Click **Find Related News** → Ranks 209,000 articles by TF-IDF cosine similarity with similarity bars and shared term explanations.

---

## 📄 Licence
Academic Project — For educational and research purposes.
