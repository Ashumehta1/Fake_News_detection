# 📰 Fake News Detection using TF–IDF
Project Overview

    This project implements a Fake News Detection system to classify whether a given news article is Real or Fake.
    We use TF–IDF (Term Frequency–Inverse Document Frequency) for text vectorization and train multiple ML models to compare their performance.

⚙️ Workflow
    1. Data Collection

        Dataset contains fake and true news articles.

        data_fake → Fake news samples

        data_true → Verified / Real news samples

    2. Data Preprocessing

        Applied text cleaning through a custom function wordout(text):

        Convert text to lowercase

        Remove URLs (http, https, www)

        Remove email addresses

        Remove HTML tags (<...>)

        Remove punctuation, numbers, and special characters

        Remove extra spaces and newline characters

    3. Data Preparation

        Combined both datasets using pd.concat()

        Shuffled the dataset with frac=1 to avoid bias

        Split into training and testing sets

    4. Feature Extraction (TF–IDF)

        Used TF–IDF Vectorizer (TfidfVectorizer) to convert text into numerical features

        Each document is transformed into a vector of word importance

    5. Models Used

        Trained multiple models for comparison:

        Logistic Regression (LR)

        Decision Tree (DT)

        Gradient Boosting (GB)

        Random Forest (RDF)

    6. Evaluation

        Evaluated models using Accuracy, Precision, Recall, F1-score

        Compared which model performs best
## 📊 Results

    Logistic Regression performed best on TF–IDF features

    Ensemble methods (Random Forest, Gradient Boosting) also showed competitive results

