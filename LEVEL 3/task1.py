import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


df = pd.read_csv(r"C:\Users\Student\OneDrive - University of KwaZulu-Natal\Desktop\Data Science Folder\Data-science-intern\LEVEL 3\Sentiment dataset.csv")
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col])
df["Platform"] = df["Platform"].str.strip()
df["Sentiment"] = df["Sentiment"].str.strip()

nltk.download("stopwords")
nltk.download("punkt")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.strip()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df["Cleaned_Text"] = df["Text"].apply(clean_text)


# Reuse our sentiment grouping from EDA
positive_words = ["Positive", "Joy", "Excitement", "Contentment", "Happiness",
                  "Admiration", "Amusement", "Gratitude", "Pride", "Relief",
                  "Optimism", "Love", "Enthusiasm"]
negative_words = ["Negative", "Anger", "Sadness", "Fear", "Disgust",
                  "Frustration", "Anxiety", "Disappointment", "Guilt", "Shame",
                  "Envy", "Grief", "Hate"]

def group_sentiment(s):
    if s in positive_words:
        return "Positive"
    elif s in negative_words:
        return "Negative"
    else:
        return "Neutral"

df["Label"] = df["Sentiment"].apply(group_sentiment)

print(df[["Text", "Label"]].head(10))
print("\nLabel Distribution:")
print(df["Label"].value_counts())



print("\t\t\tBefore vs After Cleaning")
for i in range(3):
    print(f"\nOriginal : {df['Text'].iloc[i]}")
    print(f"Cleaned  : {df['Cleaned_Text'].iloc[i]}")

# Convert text to TF-IDF matrix
tfidf = TfidfVectorizer(max_features=1000)  # keep top 1000 words
X = tfidf.fit_transform(df["Cleaned_Text"])
y = df["Label"]

print(f"TF-IDF Matrix Shape: {X.shape}")
print(f"Each text is now represented as {X.shape[1]} numbers")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size: {X_train.shape[0]}")
print(f"Test size:  {X_test.shape[0]}")
print(f"\nTest Label Distribution:")
print(y_test.value_counts())

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight="balanced"  
    )
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    results[name] = round(f1, 3)
    print(f"\n{'='*40}")
    print(f"{name}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred,
          target_names=["Negative", "Neutral", "Positive"],
          zero_division=0))

# Visualization
fig, ax = plt.subplots(figsize=(8, 5))
models_names = list(results.keys())
f1_scores = list(results.values())
bars = ax.bar(models_names, f1_scores, 
              color=["#3498db", "#2ecc71"],
              edgecolor="black", width=0.4)
ax.set_title("NLP Model F1 Score Comparison")
ax.set_ylabel("Weighted F1 Score")
ax.set_ylim(0, 1)
for bar, score in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            str(score), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("nlp_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n\t\t\tF1 Score Comparison")
for name, score in results.items():
    print(f"{name:25} -> F1: {score}")