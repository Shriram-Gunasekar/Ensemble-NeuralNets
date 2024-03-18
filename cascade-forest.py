import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to calculate Renyi entropy
def renyi_entropy(probs, alpha=2):
    probs_powered = np.power(probs, alpha)
    return -np.log(np.sum(probs_powered)) / (1 - alpha)

# Initialize lists to store the predictions and misclassified samples
cascade_predictions = []
misclassified_samples = X_train.copy()

# Initialize the cascade forest ensemble
num_cascade_forests = 3
for i in range(num_cascade_forests):
    # Train a random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(misclassified_samples, y_train)
    
    # Make predictions on the entire training set
    y_pred = rf.predict(X_train)
    
    # Calculate Renyi entropy for each sample
    class_probs = rf.predict_proba(X_train)
    entropies = np.apply_along_axis(renyi_entropy, axis=1, arr=class_probs)
    
    # Identify misclassified samples based on Renyi entropy
    misclassified_indices = np.where(y_train != y_pred)[0]
    misclassified_samples = X_train[misclassified_indices]
    
    # Store the predictions of the current forest
    cascade_predictions.append(y_pred)

# Combine predictions from all cascade forests using a voting scheme
ensemble_predictions = np.mean(cascade_predictions, axis=0).round().astype(int)

# Evaluate the ensemble on the test set
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print(f'Ensemble accuracy: {ensemble_accuracy * 100:.2f}%')
