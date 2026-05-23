from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

csv_path = Path(__file__).resolve().parent / "sentiment dataset.csv"
df = pd.read_csv(csv_path)

#drop unnamed columns
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col])

# ── Fix Platform whitespace ─────────────────────────────────────
df["Platform"] = df["Platform"].str.strip()
df["Country"]  = df["Country"].str.strip()

# ── Group Sentiment into 3 buckets ─────────────────────────────
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

df["Sentiment_Group"] = df["Sentiment"].apply(group_sentiment)

# ── Step 2: Summary Statistics ──────────────────────────────────
print("=== Summary Statistics ===")
print(df[["Retweets", "Likes"]].describe().round(2))
print("\nSentiment Distribution:")
print(df["Sentiment_Group"].value_counts())
print("\nPlatform Distribution:")
print(df["Platform"].value_counts())

# ── Step 3: Visualizations ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Sentiment Dataset EDA", fontsize=16, fontweight="bold")

# 1. Sentiment distribution bar chart
df["Sentiment_Group"].value_counts().plot(kind="bar", ax=axes[0,0],
    color=["#2ecc71","#e74c3c","#3498db"], edgecolor="black")
axes[0,0].set_title("Sentiment Distribution")
axes[0,0].set_xlabel("Sentiment")
axes[0,0].set_ylabel("Count")
axes[0,0].tick_params(axis="x", rotation=0)

# 2. Likes histogram
axes[0,1].hist(df["Likes"], bins=30, color="#3498db", edgecolor="black")
axes[0,1].set_title("Distribution of Likes")
axes[0,1].set_xlabel("Likes")
axes[0,1].set_ylabel("Frequency")

# 3. Retweets histogram
axes[0,2].hist(df["Retweets"], bins=30, color="#e67e22", edgecolor="black")
axes[0,2].set_title("Distribution of Retweets")
axes[0,2].set_xlabel("Retweets")
axes[0,2].set_ylabel("Frequency")

# 4. Box plot — Likes by Sentiment
df.boxplot(column="Likes", by="Sentiment_Group", ax=axes[1,0],
           patch_artist=True)
axes[1,0].set_title("Likes by Sentiment")
axes[1,0].set_xlabel("Sentiment")
axes[1,0].set_ylabel("Likes")
plt.sca(axes[1,0])
plt.xticks(rotation=0)

# 5. Scatter plot — Likes vs Retweets
colors = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#3498db"}
for sentiment, group in df.groupby("Sentiment_Group"):
    axes[1,1].scatter(group["Likes"], group["Retweets"],
                      label=sentiment, alpha=0.5,
                      color=colors[sentiment], s=20)
axes[1,1].set_title("Likes vs Retweets")
axes[1,1].set_xlabel("Likes")
axes[1,1].set_ylabel("Retweets")
axes[1,1].legend()

# 6. Platform distribution
df["Platform"].value_counts().plot(kind="bar", ax=axes[1,2],
    color="#9b59b6", edgecolor="black")
axes[1,2].set_title("Posts by Platform")
axes[1,2].set_xlabel("Platform")
axes[1,2].set_ylabel("Count")
axes[1,2].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n Plots saved to eda_plots.png")

# ── Step 4: Correlation Matrix ──────────────────────────────────
plt.figure(figsize=(6, 4))
corr = df[["Retweets", "Likes", "Year", "Month", "Day", "Hour"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nCorrelation matrix saved to correlation_matrix.png")