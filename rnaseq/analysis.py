import pandas as pd
import numpy as np

np.random.seed(42)

genes = [f"Gene_{i}" for i in range(1, 101)]

data = pd.DataFrame({
    "Control_1": np.random.poisson(20, 100),
    "Control_2": np.random.poisson(22, 100),
    "Treatment_1": np.random.poisson(35, 100),
    "Treatment_2": np.random.poisson(38, 100),
}, index=genes)

print(data.head())
import matplotlib.pyplot as plt

# średnie ekspresje
means = data.mean()

plt.figure()
means.plot(kind="bar")
plt.title("Mean gene expression")
plt.ylabel("Expression level")
plt.tight_layout()

plt.savefig("mean_expression.png")
plt.show()
from sklearn.decomposition import PCA

# PCA działa na próbkach → transponujemy dane
X = data.T

pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.figure()

plt.scatter(result[0, 0], result[0, 1], label="Control_1")
plt.scatter(result[1, 0], result[1, 1], label="Control_2")
plt.scatter(result[2, 0], result[2, 1], label="Treatment_1")
plt.scatter(result[3, 0], result[3, 1], label="Treatment_2")

plt.legend()
plt.title("PCA of RNA-seq samples")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.savefig("pca_plot.png")
plt.show()
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt

# grupy
control = data[["Control_1", "Control_2"]]
treatment = data[["Treatment_1", "Treatment_2"]]

# log2 fold change
log2fc = np.log2(treatment.mean(axis=1) / control.mean(axis=1))

# test statystyczny (p-value)
pvals = []
for i in range(len(data)):
    stat, p = ttest_ind(treatment.iloc[i], control.iloc[i])
    pvals.append(p)

pvals = np.array(pvals)
neg_log_p = -np.log10(pvals)

# volcano plot
plt.figure()
plt.scatter(log2fc, neg_log_p, alpha=0.7)

plt.axvline(x=0, linestyle="--")
plt.xlabel("log2 Fold Change")
plt.ylabel("-log10 p-value")
plt.title("Volcano Plot (RNA-seq)")

plt.savefig("volcano_plot.png")
plt.show()
import pandas as pd
import numpy as np
def create_data():
    np.random.seed(42)

    genes = [f"Gene_{i}" for i in range(1, 101)]

    data = pd.DataFrame({
        "Control_1": np.random.poisson(20, 100),
        "Control_2": np.random.poisson(22, 100),
        "Treatment_1": np.random.poisson(35, 100),
        "Treatment_2": np.random.poisson(38, 100),
    }, index=genes)

    return data
def compute_log2fc(data):
    control = data[["Control_1", "Control_2"]].mean(axis=1)
    treatment = data[["Treatment_1", "Treatment_2"]].mean(axis=1)

    log2fc = np.log2(treatment / control)
    return log2fc
data = create_data()
log2fc = compute_log2fc(data)

print("Top genes by change:")
print(log2fc.sort_values(ascending=False).head())
top_up = log2fc.sort_values(ascending=False).head(10)
top_down = log2fc.sort_values(ascending=True).head(10)

print("\nTOP UPREGULATED GENES:")
print(top_up)

print("\nTOP DOWNREGULATED GENES:")
print(top_down)
results = pd.DataFrame({
    "log2FC": log2fc
})

results.to_csv("gene_results.csv")