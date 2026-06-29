import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(0)

data = pd.DataFrame({
    "sample1": np.random.normal(0, 1, 50),
    "sample2": np.random.normal(0, 1, 50),
    "sample3": np.random.normal(1, 1, 50),
    "sample4": np.random.normal(1, 1, 50)
})

data = data.T

pca = PCA(n_components=2)
result = pca.fit_transform(data)

plt.scatter(result[:, 0], result[:, 1])
plt.title("Simple PCA plot")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.savefig("pca_project.png")
plt.show()