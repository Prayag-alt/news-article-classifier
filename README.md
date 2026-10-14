# News Intelligence System: Multi-Category Classification, Uncertainty Estimation, and Semantic Recommendation

An end-to-end Machine Learning and Natural Language Processing (NLP) system that categorizes news articles across 42 fine-grained categories, evaluates prediction uncertainty using raw Linear Support Vector Machine (SVM) decision margins, extracts salient keywords via TF-IDF feature weights, performs rule-based sentiment profiling, and delivers personalized recommendations — including a Research-to-News retrieval pipeline matching academic research papers with historical news coverage.

Built with Python, scikit-learn, an 80,000-feature TF-IDF unigram and bigram matrix, a trained Linear SVM, and an interactive Streamlit dashboard.

---

## Project Overview

News publishers and content aggregators produce millions of articles spanning diverse domains. Manually organizing, cataloging, and cross-referencing news articles with scientific literature is resource-intensive.

This project delivers a five-pillar intelligence suite:

1. **News Classification:** Predicts one of 42 fine-grained categories using a trained Linear Support Vector Machine with One-vs-Rest (OvR) decision hyperplanes.
2. **Prediction Confidence and Uncertainty:** Employs raw geometric decision-function distances and margin analysis (gap between first and second candidate scores) to detect ambiguous classifications without fabricating posterior probabilities.
3. **Salient Keyword Extraction:** Isolates top discriminative unigram and bigram features based on fitted TF-IDF weights.
4. **Sentiment Analysis:** Analyzes emotional polarity (-1.0 to +1.0) and subjectivity (0.0 to 1.0) using TextBlob.
5. **Personalized News Recommendations:**
   - **Interest-Based Feed (Mode A):** Filters, samples, and orders articles by user-selected categories and publication date.
   - **Research-to-News Retrieval (Mode B):** Ingests academic abstracts or uploaded PDF/TXT research documents, vectorizes text with TF-IDF, computes cosine similarities across the 209,527-article corpus, and returns matching coverage with similarity progress indicators and shared-term explanations.

---

## Dataset Specifications

The project utilizes the **HuffPost News Category Dataset v3** (Rishabh Misra, Kaggle):

| Property | Value |
|---|---:|
| Total Articles | 209,527 |
| Valid Preprocessed Articles | 209,034 |
| Target Categories | 42 |
| Temporal Coverage | January 28, 2012 to September 23, 2022 |
| Vocabulary / Feature Space | 80,000 (Unigrams + Bigrams) |
| Dominant Category | POLITICS (35,602 articles) |
| Smallest Category | EDUCATION (1,014 articles) |
| Class Imbalance Ratio | ~35.11:1 |

> **Historical Dataset Disclosure:** The dataset spans 2012 to 2022. The system provides historical news contextualization and does not connect to live, real-time news feeds.

---

## Baseline Model Performance & Evaluation

The system was evaluated against standard classification baselines on the multi-class test partition. The trained Linear SVM weights and TF-IDF matrix are preserved directly from these experiments:

### Model Benchmark Comparison
| Model Architecture | Accuracy | Macro F1-Score | Status |
|---|---:|---:|---|
| Multinomial Naive Bayes | 44.53% | 0.1653 | Baseline |
| Logistic Regression (L2) | 60.83% | 0.4497 | Strong Baseline |
| **Linear SVM (Selected)** | **61.29%** | **0.4824** | **Production Selected** |

### Selected Linear SVM Performance Profile
| Metric | Measured Value |
|---|---:|
| **Accuracy** | **61.29%** |
| **Macro Precision** | **52.63%** |
| **Macro Recall** | **45.92%** |
| **Macro F1-Score** | **0.4824** |
| **Weighted F1-Score** | **0.5976** |
| Regularization Parameter C | 1.0 |
| Loss Function | Squared Hinge |
| Multi-class Strategy | One-vs-Rest (OvR) |

---

## Google Colab Notebook

The complete data pipeline, exploratory data analysis, visualizations, model benchmarking, and five core functionalities are available in the accompanying Jupyter notebook:

- **Notebook File:** `news_intelligence_system.ipynb`
- **Contents:**
  - Concise Markdown headings and cell structure.
  - Figure 1: News Category Frequency Distribution (Bar chart highlighting ~35:1 class imbalance).
  - Figure 2: Article Text Length Distribution (Word and character counts).
  - Figure 3: Benchmark Model Comparison Across Architectures (Grouped bar chart for Accuracy & Macro F1).
  - Figure 4: Linear SVM Normalized Confusion Matrix (Heatmap across representative categories).
  - End-to-end implementation of all five functionalities.
- **Run in Google Colab:**
  1. Open [Google Colab](https://colab.research.google.com/).
  2. Select **GitHub** tab and enter: `Prayag-alt/news-article-classifier`.
  3. Select `news_intelligence_system.ipynb`.
  4. Run all cells sequentially.

---

## Quick Start & Local Execution

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11
- pip package manager

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Prayag-alt/news-article-classifier.git
cd news-article-classifier

pip install -r requirements.txt
```

### 3. Model and Dataset Files
Ensure the following files are present in the project directory:
- `news_category_svm.pkl`: Trained Linear SVM model file (~26 MB)
- `news_category_tfidf.pkl`: Fitted TF-IDF vectorizer (~3 MB)
- `News_Category_Dataset_v3.json.gz`: Compressed dataset file (~27 MB)

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Deployment to Streamlit Community Cloud

The repository is configured for immediate deployment on Streamlit Community Cloud:

1. **GitHub Repository:** Ensure the repository is pushed to `main` branch at `https://github.com/Prayag-alt/news-article-classifier`.
2. **Streamlit Cloud Setup:**
   - Navigate to [share.streamlit.io](https://share.streamlit.io).
   - Sign in with your GitHub account.
   - Click **Create app** / **New app**.
   - Set **Repository:** `Prayag-alt/news-article-classifier`
   - Set **Branch:** `main`
   - Set **Main file path:** `app.py`
   - Click **Deploy!**
3. **Execution:** Streamlit Cloud installs dependencies from `requirements.txt`, loads the cached SVM model and TF-IDF vectorizer, and launches the application.

---

## Repository Structure

```
news-article-classifier/
|--  .streamlit/
|   \--  config.toml               # Restrained navy/blue Streamlit theme configuration
|--  app.py                        # Professional Streamlit dashboard (Zero emojis)
|--  news_classifier.py            # Model loading, classification, and sentiment analysis
|--  text_processing.py            # Text preprocessing, keyword extraction, and document parsers
|--  recommendation.py             # Interest-based feed and Research-to-News recommendation engine
|--  news_intelligence_system.ipynb # Complete academic Google Colab notebook with figures
|--  requirements.txt              # Production Python dependencies
|--  .gitignore                    # Git ignore specifications
|--  README.md                     # Scholarly project documentation
|--  news_category_svm.pkl         # Trained Linear SVM model file
|--  news_category_tfidf.pkl       # Fitted TF-IDF vectorizer
\--  News_Category_Dataset_v3.json.gz # Compressed news corpus (209,527 articles)
```

---

## System Verification Checklist

- [x] **Feature 1: News Classification**
  - Accurate categorization across 42 classes using trained Linear SVM.
  - Interactive test samples for Technology, Politics, Sports, and Science.
- [x] **Feature 2: Prediction Confidence & Uncertainty**
  - Decision margin between top two scores determines High, Moderate, or Ambiguous confidence.
  - Displays raw signed hyperplane distances without fabricated probabilities.
- [x] **Feature 3: Keyword Extraction**
  - Extracts 5 to 10 discriminating keywords with fitted TF-IDF weights.
- [x] **Feature 4: Sentiment Analysis**
  - Returns polarity (-1.0 to +1.0) and subjectivity (0.0 to 1.0) with clean status tags.
- [x] **Feature 5A: Personalized News Feed**
  - Category multi-select and topic presets with publication date ordering.
- [x] **Feature 5B: Research-to-News Retrieval**
  - Dual tabs for abstract text pasting and PDF/TXT upload (with PyMuPDF parsing).
  - Cosine similarity ranking across 209k articles with similarity bars and shared term rationale.
- [x] **Aesthetics & UI/UX Standards**
  - Zero emojis across the entire interface.
  - Restrained navy, blue, and white color palette with accessible contrast.
  - Clean card layouts, informative loading states, and robust error handling.

---

## License & Academic Disclaimer
Academic Project — For educational and research purposes.
The dataset is licensed by HuffPost / Kaggle (Rishabh Misra).
