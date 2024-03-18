import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the GELU activation function
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the first autoencoder with GELU activation
autoencoder1 = Sequential([
    Dense(4, input_dim=4, activation=gelu),
    Dense(2, activation=gelu),
    Dense(4, activation='linear')  # Output layer with linear activation
])

autoencoder1.compile(optimizer=Adam(), loss='mse')

# Train the first autoencoder
autoencoder1.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=16, verbose=0)

# Extract the encoded representations from the first autoencoder
encoder1 = Sequential(autoencoder1.layers[:2])  # Extract encoder layers

# Encode the training and testing data using the first encoder
X_train_encoded1 = encoder1.predict(X_train_scaled)
X_test_encoded1 = encoder1.predict(X_test_scaled)

# Build the second autoencoder with GELU activation
autoencoder2 = Sequential([
    Dense(2, input_dim=2, activation=gelu),
    Dense(1, activation='linear')  # Output layer with linear activation
])

autoencoder2.compile(optimizer=Adam(), loss='mse')

# Train the second autoencoder
autoencoder2.fit(X_train_encoded1, X_train_encoded1, epochs=50, batch_size=16, verbose=0)

# Extract the encoded representations from the second autoencoder
encoder2 = Sequential(autoencoder2.layers[:1])  # Extract encoder layer

# Encode the training and testing data using the second encoder
X_train_encoded2 = encoder2.predict(X_train_encoded1)
X_test_encoded2 = encoder2.predict(X_test_encoded1)

# Build a simple classifier on top of the encoded representations
classifier = Sequential([
    Dense(10, input_dim=1, activation=gelu),
    Dense(3, activation='softmax')  # Output layer for classification
])

classifier.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the classifier on the encoded representations
classifier.fit(X_train_encoded2, y_train, epochs=50, batch_size=16, verbose=0)

# Evaluate the model on the test set
loss, accuracy = classifier.evaluate(X_test_encoded2, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')
