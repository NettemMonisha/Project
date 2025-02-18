# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# ================================================
# Task (a): PCA for Dimensionality Reduction
# ================================================

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1e-8  # Prevent division by zero
    return (X - mean) / std


def compute_covariance_matrix(X):
    return np.cov(X, rowvar=False)


def pca(X, n_components):
    X_std = standardize(X)
    cov_matrix = compute_covariance_matrix(X_std)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
    return np.dot(X_std, top_eigenvectors)


# Generate synthetic digits dataset
np.random.seed(42)
X_digits = np.random.rand(1797, 64)
y_digits = np.random.randint(0, 10, 1797)

# Apply PCA to reduce to 2 components
X_pca = pca(X_digits, 2)


# ================================================
# Task (b): K-means Clustering
# ================================================

def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


# Apply K-means clustering
k = 10
labels_kmeans, _ = kmeans(X_pca, k)


# ================================================
# Task (c): Gaussian Mixture Model (GMM) Clustering
# ================================================

class ManualGMM:
    def __init__(self, k, max_iters=100):
        self.k, self.max_iters = k, max_iters

    def initialize(self, X):
        np.random.seed(42)
        self.n, self.d = X.shape
        self.means = X[np.random.choice(self.n, self.k, replace=False)]
        self.covariances = np.array([np.eye(self.d) for _ in range(self.k)])
        self.weights = np.ones(self.k) / self.k

    def gaussian_pdf(self, X, mean, cov):
        d = X.shape[1]
        coef = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(cov) ** 0.5)
        exponent = -0.5 * np.sum((X - mean) @ np.linalg.inv(cov) * (X - mean), axis=1)
        return coef * np.exp(exponent)

    def e_step(self, X):
        self.resp = np.zeros((self.n, self.k))
        for i in range(self.k):
            self.resp[:, i] = self.weights[i] * self.gaussian_pdf(X, self.means[i], self.covariances[i])
        self.resp /= self.resp.sum(axis=1, keepdims=True)

    def m_step(self, X):
        Nk = self.resp.sum(axis=0)
        self.weights = Nk / self.n
        self.means = (self.resp.T @ X) / Nk[:, np.newaxis]
        for i in range(self.k):
            diff = X - self.means[i]
            self.covariances[i] = (self.resp[:, i, np.newaxis] * diff).T @ diff / Nk[i]

    def fit(self, X):
        self.initialize(X)
        for _ in range(self.max_iters):
            self.e_step(X)
            self.m_step(X)

    def predict(self, X):
        self.e_step(X)
        return np.argmax(self.resp, axis=1)


# Apply GMM clustering
gmm_digits = ManualGMM(k=10)
gmm_digits.fit(X_pca)
labels_gmm = gmm_digits.predict(X_pca)


# ================================================
# Task (d): Compute Silhouette Score
# ================================================

def silhouette_score_manual(X, labels):
    """Calculate silhouette score manually."""
    unique_clusters = np.unique(labels)
    silhouette_scores = []

    for i in range(len(X)):
        same_cluster = X[labels == labels[i]]
        other_clusters = [X[labels == c] for c in unique_clusters if c != labels[i]]

        a_i = np.mean(np.linalg.norm(same_cluster - X[i], axis=1)) if len(same_cluster) > 1 else 0
        b_i = min(
            [np.mean(np.linalg.norm(cluster - X[i], axis=1)) for cluster in other_clusters]) if other_clusters else 0

        silhouette_scores.append((b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0)

    return np.mean(silhouette_scores)


# Compute silhouette scores for K-Means and GMM on Digits dataset
kmeans_silhouette_digits = silhouette_score_manual(X_pca, np.array(labels_kmeans))
gmm_silhouette_digits = silhouette_score_manual(X_pca, np.array(labels_gmm))


# ================================================
# Task (e): Apply to Synthetic Blob Dataset
# ================================================

def make_blobs(n_samples=1000, centers=4, cluster_std=1.0):
    np.random.seed(42)
    X = np.vstack([np.random.randn(n_samples // centers, 2) * cluster_std + np.random.uniform(-10, 10, size=2)
                   for _ in range(centers)])
    y = np.repeat(range(centers), n_samples // centers)
    return X, y


X_blob, y_blob = make_blobs(n_samples=1000, centers=4)

# Apply K-means to blob dataset
labels_kmeans_blob, _ = kmeans(X_blob, k=4)

# Apply GMM to blob dataset
gmm_blob = ManualGMM(k=4)
gmm_blob.fit(X_blob)
labels_gmm_blob = gmm_blob.predict(X_blob)

# Compute silhouette scores for K-Means and GMM on Blob dataset
kmeans_silhouette_blob = silhouette_score_manual(X_blob, np.array(labels_kmeans_blob))
gmm_silhouette_blob = silhouette_score_manual(X_blob, np.array(labels_gmm_blob))


# ================================================
# Task (f): Generate Graphs
# ================================================

def plot_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', alpha=0.5)
    plt.title(title)
    plt.xlabel("PCA Component 1" if X.shape[1] == 2 else "Feature 1")
    plt.ylabel("PCA Component 2" if X.shape[1] == 2 else "Feature 2")
    plt.colorbar(label="Cluster")
    plt.show()


# Plot clustering results
plot_clusters(X_pca, labels_kmeans, "K-Means Clustering on PCA-Reduced Digits Data")
plot_clusters(X_pca, labels_gmm, "GMM Clustering on PCA-Reduced Digits Data")
plot_clusters(X_blob, labels_kmeans_blob, "K-Means Clustering on Blob Dataset")
plot_clusters(X_blob, labels_gmm_blob, "GMM Clustering on Blob Dataset")

# ================================================
# Task (g): Display Scores
# ================================================

print("\nSilhouette Scores for Clustering Performance:")
print(f"K-Means Silhouette Score (Digits Dataset): {kmeans_silhouette_digits:.4f}")
print(f"GMM Silhouette Score (Digits Dataset): {gmm_silhouette_digits:.4f}")
print(f"K-Means Silhouette Score (Blob Dataset): {kmeans_silhouette_blob:.4f}")
print(f"GMM Silhouette Score (Blob Dataset): {gmm_silhouette_blob:.4f}")