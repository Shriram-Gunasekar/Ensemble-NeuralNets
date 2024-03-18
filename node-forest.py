import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torchdiffeq import odeint
import torch
import torch.nn as nn

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple Neural ODE model for feature extraction
class NeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralODE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, t, x):
        dx_dt = torch.sigmoid(self.fc1(x))
        return self.fc2(dx_dt)

# Initialize and train the Neural ODE model (this is a simplified example)
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 32
model = NeuralODE(input_dim, hidden_dim, output_dim)

# Define a function to extract features using the trained Neural ODE model
def extract_features(data):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    features = odeint(model, data_tensor, torch.tensor([0, 1]))[-1].detach().numpy()
    return features

# Extract features for training and testing data
X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_features, y_train)

# Evaluate the Random Forest classifier on the test set
accuracy = rf_classifier.score(X_test_features, y_test)
print(f'Random Forest Classifier Accuracy: {accuracy * 100:.2f}%')
