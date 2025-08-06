import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("benchmark_results.csv")

# Set style
sns.set_theme(style="whitegrid", context="talk")

# Ensure DB order stays consistent
db_order = ["Milvus", "Qdrant", "FAISS", "pgvector"]

# <<<<< 1. Insert Times >>>>>
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="Size", y="Insert Time (s)", hue="DB", marker="o", hue_order=db_order)
plt.title("Insert Time vs Dataset Size")
plt.ylabel("Insert Time (seconds)")
plt.xlabel("Dataset Size (# vectors)")
plt.legend(title="Database")
plt.tight_layout()
plt.show()

# <<<<< 2. Query Times >>>>>
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

# <<<<< 2a. Query TopK Times >>>>>
sns.lineplot(
    data=df, x="Size", y="Avg Query TopK Time (s)",
    hue="DB", marker="o", hue_order=db_order, ax=axes[0]
)
axes[0].set_title("TopK Query Time")
axes[0].set_ylabel("Avg Time (s)")
axes[0].set_xlabel("Dataset Size (# vectors)")

# <<<<< 2b. Query Filter Times >>>>>
sns.lineplot(
    data=df, x="Size", y="Avg Query Filter Time (s)",
    hue="DB", marker="o", hue_order=db_order, ax=axes[1]
)
axes[1].set_title("Filter Query Time")
axes[1].set_ylabel("Avg Time (s)")
axes[1].set_xlabel("Dataset Size (# vectors)")

# <<<<< 2c. Query Range Times >>>>>
sns.lineplot(
    data=df, x="Size", y="Avg Query Range Time (s)",
    hue="DB", marker="o", hue_order=db_order, ax=axes[2]
)
axes[2].set_title("Range Query Time")
axes[2].set_ylabel("Avg Time (s)")
axes[2].set_xlabel("Dataset Size (# vectors)")

handles, labels = axes[0].get_legend_handles_labels()
for ax in axes:
    ax.get_legend().remove()
fig.legend(handles, labels, title="Database", loc="upper center", ncol=4)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# <<<<< 3. Update Times >>>>>
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Size", y="Avg Update Time (s)", hue="DB")
plt.title("Average Update Time vs Dataset Size")
plt.ylabel("Average Update Time (seconds)")
plt.xlabel("Dataset Size")
plt.yscale("log")
plt.tight_layout()
plt.show()

# <<<<< 4. Delete Times >>>>>
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Size", y="Avg Delete Time (s)", hue="DB")
plt.title("Average Delete Time vs Dataset Size")
plt.ylabel("Average Delete Time (seconds)")
plt.xlabel("Dataset Size")
plt.yscale("log")
plt.tight_layout()
plt.show()

# <<<<< 5. Heatmap for Average Query TopK Time >>>>>
pivot = df.pivot(index="DB", columns="Size", values="Avg Query TopK Time (s)")

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="coolwarm")
plt.title("Avg Query TopK Time Heatmap")
plt.ylabel("Database")
plt.xlabel("Dataset Size")
plt.show()
