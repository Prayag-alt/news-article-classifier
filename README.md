\# 📰 News Article Categorization and Classification



A Machine Learning based Natural Language Processing (NLP) project that automatically classifies news articles into one of 42 predefined categories using TF-IDF feature extraction and a Linear Support Vector Machine (SVM).



\## 📌 Project Overview



News websites publish a large number of articles every day across different topics. Manually categorizing these articles is time-consuming.



This project develops an automated news classification system that takes a news article as input and predicts its category.



The project includes:



\- Dataset exploration and analysis

\- Text preprocessing

\- Exploratory Data Analysis (EDA)

\- TF-IDF feature extraction

\- Multiple machine learning models

\- Hyperparameter tuning

\- Error and confusion analysis

\- Class imbalance analysis

\- Model comparison

\- Streamlit web application

\- Model deployment package



\## 📊 Dataset



The project uses the \*\*HuffPost News Category Dataset v3\*\*.



\### Dataset Statistics



| Property | Value |

|---|---:|

| Total articles | 209,527 |

| Articles after preprocessing | 209,034 |

| Number of categories | 42 |

| Training samples | 167,227 |

| Testing samples | 41,807 |

| Train/Test split | 80/20 |

| Date range | 2012-01-28 to 2022-09-23 |



The dataset contains the following main fields:



\- `headline`

\- `short\_description`

\- `category`

\- `authors`

\- `date`

\- `link`



\## 🔍 Exploratory Data Analysis



EDA was performed to understand:



\- Category distribution

\- Class imbalance

\- Headline length

\- Description length

\- Most frequent words

\- Dataset quality and duplicates



The largest category is \*\*POLITICS\*\* with 35,602 articles, while \*\*EDUCATION\*\* is the smallest with 1,014 articles.



This gives an imbalance ratio of approximately \*\*35.11:1\*\*.



\## 🧹 Text Preprocessing



The following preprocessing steps were applied:



1\. Combined headline and short description

2\. Converted text to lowercase

3\. Removed URLs

4\. Removed punctuation and special characters

5\. Removed English stopwords

6\. Removed very short words

7\. Removed exact duplicate records

8\. Removed empty records after preprocessing



Final dataset size after preprocessing:



\*\*209,034 articles\*\*



\## 🧮 Feature Extraction



TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert text into numerical features.



Configuration:



\- Maximum features: \*\*80,000\*\*

\- N-gram range: \*\*Unigrams + Bigrams\*\*

\- Minimum document frequency: \*\*2\*\*

\- Maximum document frequency: \*\*0.95\*\*

\- Sublinear TF: Enabled



The TF-IDF vectorizer was fitted only on the training data to avoid data leakage.



\## 🤖 Machine Learning Models



Three baseline classification models were evaluated:



1\. Multinomial Naive Bayes

2\. Logistic Regression

3\. Linear Support Vector Machine (SVM)



\### Model Comparison



| Model | Accuracy | Macro F1 |

|---|---:|---:|

| Naive Bayes | 44.53% | 16.53% |

| Logistic Regression | 60.83% | 44.97% |

| Linear SVM | \*\*61.29%\*\* | \*\*48.24%\*\* |



Linear SVM achieved the best overall performance.



\## ⚙️ Hyperparameter Tuning



Linear SVM was evaluated with different values of `C`:



\- 0.5

\- 1.0

\- 1.5

\- 2.0



A stratified subset of the training data was used for efficient tuning.



The selected value was:



\*\*C = 1.0\*\*



\## 🏆 Final Model



The final model is a \*\*Linear SVM with TF-IDF features\*\*.



\### Final Performance



| Metric | Score |

|---|---:|

| Accuracy | \*\*61.29%\*\* |

| Macro Precision | \*\*52.63%\*\* |

| Macro Recall | \*\*45.92%\*\* |

| Macro F1 | \*\*48.24%\*\* |

| Weighted F1 | \*\*59.76%\*\* |



\## ⚖️ Class Imbalance Analysis



A class-balanced Linear SVM was also tested.



| Model | Accuracy | Macro F1 | Macro Recall |

|---|---:|---:|---:|

| Original Linear SVM | \*\*61.29%\*\* | \*\*48.24%\*\* | 45.92% |

| Balanced Linear SVM | 59.08% | 48.10% | \*\*50.23%\*\* |



Although class balancing improved macro recall, it slightly reduced Macro F1 and accuracy. Therefore, the original Linear SVM was selected as the final model.



\## 🔎 Error Analysis



Some categories are difficult to distinguish because their vocabulary and topics overlap.



Important confusion patterns included:



\- HEALTHY LIVING → WELLNESS

\- PARENTS → PARENTING

\- TASTE → FOOD \& DRINK

\- COMEDY → ENTERTAINMENT

\- STYLE → STYLE \& BEAUTY

\- BUSINESS → POLITICS



These errors demonstrate the challenge of classifying semantically similar news categories.



\## 🌐 Streamlit Application



A Streamlit web application was developed for real-time prediction.



The user can enter a news article or headline and the application returns the predicted category.



\### Application Pipeline



```text

User Input

&#x20;   ↓

Text Preprocessing

&#x20;   ↓

TF-IDF Vectorization

&#x20;   ↓

Linear SVM

&#x20;   ↓

Predicted News Category

