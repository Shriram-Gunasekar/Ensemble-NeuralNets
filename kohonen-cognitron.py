import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Scale the features to [0, 1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the SOM grid size
grid_size = (10, 10)  # 10x10 grid

# Initialize the SOM
som = MiniSom(grid_size[0], grid_size[1], X_scaled.shape[1], sigma=1.0, learning_rate=0.5)

# Train the SOM
num_epochs = 100
som.train_random(X_scaled, num_epochs)

# Visualize the SOM
plt.figure(figsize=(8, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Plot distance map
plt.colorbar()

# Mark the data points on the map
markers = ['o', 's', 'D']  # Marker shapes for different classes
colors = ['r', 'g', 'b']  # Colors for different classes
for i, x in enumerate(X_scaled):
    winner = som.winner(x)  # Find the winning neuron for the input
    plt.plot(winner[0] + 0.5, winner[1] + 0.5, markers[y[i]], markerfacecolor='None',
             markeredgecolor=colors[y[i]], markersize=10, markeredgewidth=2)

plt.title('Kohonen Feature Map (Self-Organizing Map)')
plt.show()

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Split the dataset into training and testing sets using the SOM representations
X_som = np.array([som.winner(x) for x in X_scaled])  # SOM representations
X_train, X_test, y_train, y_test = train_test_split(X_som, y, test_size=0.2, random_state=42)

# Define the Cognitron network
cognitron = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # Output layer for 3 classes in Iris dataset
])

cognitron.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Cognitron network
num_epochs_cognitron = 50
cognitron.fit(X_train, y_train, epochs=num_epochs_cognitron, batch_size=16, verbose=1)

# Evaluate the Cognitron network on the test set
loss, accuracy = cognitron.evaluate(X_test, y_test)
print(f'Cognitron Test Accuracy: {accuracy * 100:.2f}%')
