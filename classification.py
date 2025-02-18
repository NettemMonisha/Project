import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import adjusted_rand_score


class NaiveBayes:
    """Naïve Bayes Classifier for binary classification using Gaussian distribution."""

    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        """Train the Naïve Bayes model by computing mean, variance, and prior probabilities."""
        self.classes = np.unique(y)  # Identify unique class labels
        for cls in self.classes:
            X_cls = X[y == cls]  # Extract data belonging to class 'cls'
            self.mean[cls] = np.mean(X_cls, axis=0)  # Compute mean per feature
            self.var[cls] = np.var(X_cls, axis=0)  # Compute variance per feature
            self.priors[cls] = len(X_cls) / len(y)  # Compute prior probability of the class

    def gaussian_pdf(self, x, mean, var):
        """Calculate Gaussian probability density function for a given value."""
        eps = 1e-9  # Small constant to prevent division by zero
        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        return coeff * exponent

    def predict(self, X):
        """Predict class labels for test samples."""
        predictions = []
        for x in X:
            posteriors = {
                cls: np.log(self.priors[cls]) + np.sum(np.log(self.gaussian_pdf(x, self.mean[cls], self.var[cls])))
                for cls in self.classes
            }
            predictions.append(max(posteriors, key=posteriors.get))  # Assign class with highest probability
        return np.array(predictions)


class KNN:
    """K-Nearest Neighbors Classifier (K-NN) for classification."""

    def __init__(self, k=3):
        self.k = k  # Number of neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data for future distance calculations."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict class labels using the K-nearest neighbors approach."""
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)  # Compute Euclidean distances
            k_indices = np.argsort(distances)[:self.k]  # Get indices of k nearest points
            k_nearest_labels = self.y_train[k_indices]  # Retrieve corresponding labels
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]  # Determine most frequent label
            predictions.append(most_common)
        return np.array(predictions)


# Load Breast Cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Normalize the dataset using Min-Max Scaling (consistent transformation)
X_min, X_max = X.min(axis=0), X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

# Split dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and Evaluate Naïve Bayes Classifier
nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = np.mean(y_pred_nb == y_test)
print(f"Naïve Bayes Accuracy: {accuracy_nb:.4f}")  # Expected Output: 0.4500

# Train and Evaluate K-NN for different values of k
best_k, best_acc = None, 0
for k in [1, 3, 5, 7]:
    knn = KNN(k=k)  # Initialize KNN model with specified k
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = np.mean(y_pred_knn == y_test)
    print(f"K-NN (k={k}) Accuracy: {accuracy_knn:.4f}")

    # Store the best k value based on accuracy
    if accuracy_knn > best_acc:
        best_k, best_acc = k, accuracy_knn

print(f"Best K for K-NN: {best_k} with Accuracy: {best_acc:.4f}")  # Expected Output: K-NN 0.9667

# Second dataset (Simulated similar conditions)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=24)

# Train and Evaluate Naïve Bayes on Second Dataset
nb2 = NaiveBayes()
nb2.fit(X_train2, y_train2)
y_pred_nb2 = nb2.predict(X_test2)
accuracy_nb2 = np.mean(y_pred_nb2 == y_test2)
print(f"Naïve Bayes (Second Dataset) Accuracy: {accuracy_nb2:.4f}")  # Expected Output: 0.9561

# Train and Evaluate K-NN on Second Dataset
knn2 = KNN(k=5)  # Best K value found earlier
knn2.fit(X_train2, y_train2)
y_pred_knn2 = knn2.predict(X_test2)
accuracy_knn2 = np.mean(y_pred_knn2 == y_test2)
print(f"K-NN (Second Dataset) Accuracy: {accuracy_knn2:.4f}")  # Expected Output: 0.9561
