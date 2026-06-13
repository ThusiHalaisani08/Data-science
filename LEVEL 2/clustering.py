import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv(r"C:\Users\Student\OneDrive - University of KwaZulu-Natal\Desktop\Data Science Folder\Data-science-intern\LEVEL 2\churn-bigml-80.csv")
# Data Preprocessing

def preprocessing(df):
    df = df.copy()
    df["International plan"] = df["International plan"].map({"Yes": 1, "No": 0})
    df["Voice mail plan"] = df["Voice mail plan"].map({"Yes": 1, "No": 0})
    le = LabelEncoder()
    df["State"] = le.fit_transform(df["State"])
    df["Churn"] = df["Churn"].astype(int)
    return df

df = preprocessing(df)

x = df.drop("Churn", axis=1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

print(f"Scaled features shape: {x_scaled.shape}")

print("\t\t\t Elbow method")
inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x_scaled)
    inertias.append(kmeans.inertia_)    
    print(f"   K={k} | Inertia: {kmeans.inertia_:.1f}")

plt.figure(figsize=(8, 5))
plt.plot(k_range , inertias, "bo-", linewidth=2, markersize=8)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.xticks(k_range)
plt.tight_layout()
plt.savefig("elbow method.png",dpi=150,bbox_inches="tight")
plt.show()

print("\t\t\t Silhouette scores")
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(x_scaled)
    score = silhouette_score(x_scaled, labels)
    silhouette_scores.append(score)
    print(f"   K={k} | Silhouette Score: {score:.4f}")

best_k = k_range[np.argmax(silhouette_scores)]
print(f"\nBest k based on silhouette score: {best_k}")

plt.figure(figsize=(8,5))
plt.plot(k_range, silhouette_scores, "go-", linewidth=2, markersize=8)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal k")
plt.xticks(k_range)
plt.tight_layout()
plt.savefig("silhouette analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print("\t\t\t Final KMeans with k=2")
k_final = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"]= k_final.fit_predict(x_scaled)

print("\nCluster Distribution:")
print(df["Cluster"].value_counts().sort_index())

print("\t\t\t Cluster Profiles")
profile = df.groupby("Cluster")[["Total day minutes",
                                  "Total day charge",
                                  "Customer service calls",
                                  "International plan",
                                  "Churn"]].mean().round(3)
print(profile)

#visualisation of clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_scaled)
explained = pca.explained_variance_ratio_

print(f"\nPC1 explains: {explained[0]*100:.1f}% of variance")
print(f"PC2 explains: {explained[1]*100:.1f}% of variance")
print(f"Total explained: {sum(explained)*100:.1f}%")

plt.figure(figsize=(9, 6))
colors = ["#1f77b4", "#ff7f0e"]

for i in range(2):
    mask = df["Cluster"] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                label=f"Cluster {i} ({mask.sum()})", 
                alpha=0.5,
                color = colors[i],
                s=20)
    
plt.title("KMeans Clusters Visualized with PCA (K=2)")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}%) Varience Explained")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}%) Varience Explained")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans clusters pca.png", dpi=150, bbox_inches="tight")
plt.show()