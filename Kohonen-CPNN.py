import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Scale the features to [0, 1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the SOM grid size
grid_size = (10, 10)  # 10x10 grid

# Initialize and train the SOM
som = MiniSom(grid_size[0], grid_size[1], X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.train_random(X_scaled, 100)

# Function to find the closest neuron (BMU) in the SOM
def find_best_matching_unit(x):
    bmus = np.array([som.winner(x) for x in X_scaled])
    return bmus

# Function to initialize the weights for the CPNN
def initialize_weights(input_dim, output_dim):
    return np.random.rand(input_dim, output_dim)

# Function to update weights in the CPNN
def update_weights(weights, input_vec, output_vec, alpha):
    diff = input_vec[:, None] - weights  # Calculate differences between input vector and weights
    dists = np.linalg.norm(diff, axis=0)  # Calculate Euclidean distances
    winner_idx = np.argmin(dists)  # Find the index of the winner neuron
    weights[:, winner_idx] += alpha * (input_vec - weights[:, winner_idx])  # Update winner neuron's weights
    return weights

# Function to train the CPNN
def train_cpnn(X_train, y_train, input_dim, output_dim, epochs=100, alpha=0.1):
    weights = initialize_weights(input_dim, output_dim)
    for _ in range(epochs):
        for input_vec, target_vec in zip(X_train, y_train):
            input_vec = input_vec.reshape(-1, 1)
            target_vec = target_vec.reshape(-1, 1)
            weights = update_weights(weights, input_vec, target_vec, alpha)
    return weights

# Map classes to target vectors for CPNN training
target_vectors = np.eye(len(np.unique(y)))  # One-hot encoding for target vectors

# Find the best matching units (BMUs) for each input in the SOM
bmus = find_best_matching_unit(X_scaled)

# Train the CPNN using BMUs as inputs and target vectors as outputs
input_dim = grid_size[0] * grid_size[1]
output_dim = len(np.unique(y))
weights_cpnn = train_cpnn(bmus, target_vectors[y], input_dim, output_dim)

# Function to predict using the CPNN
def predict_cpnn(weights, input_vec):
    input_vec = input_vec.reshape(-1, 1)
    diff = input_vec[:, None] - weights  # Calculate differences between input vector and weights
    dists = np.linalg.norm(diff, axis=0)  # Calculate Euclidean distances
    winner_idx = np.argmin(dists)  # Find the index of the winner neuron
    return winner_idx

# Test the CPNN on a new input
new_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # New input data point
scaled_input = scaler.transform(new_input)
bmu_new_input = find_best_matching_unit(scaled_input)[0]  # Find BMU for the new input
predicted_class_idx = predict_cpnn(weights_cpnn, bmu_new_input)

# Map predicted index to class label
predicted_class = iris.target_names[predicted_class_idx]

print(f'Predicted class for input {new_input}: {predicted_class}')
