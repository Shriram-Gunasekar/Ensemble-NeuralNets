import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features to [0, 1] range for GAN training
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the generator and discriminator models
def build_generator(input_dim, output_dim):
    generator = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='sigmoid')
    ])
    return generator

def build_discriminator(input_dim):
    discriminator = Sequential([
        Dense(64, input_dim=input_dim, activation=LeakyReLU(alpha=0.2)),
        Dense(32, activation=LeakyReLU(alpha=0.2)),
        Dense(1, activation='sigmoid')
    ])
    discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return discriminator

# Set random seed for reproducibility
np.random.seed(42)

# Build and compile the discriminator
discriminator = build_discriminator(X_train_scaled.shape[1])

# Define the GAN model
def build_gan(generator, discriminator):
    gan = Sequential([generator, discriminator])
    gan.compile(optimizer=Adam(), loss='binary_crossentropy')
    return gan

# Build the generator and GAN models
generator = build_generator(X_train_scaled.shape[1], X_train_scaled.shape[1])
gan = build_gan(generator, discriminator)

# Train the GAN model
epochs = 100
batch_size = 16

for epoch in range(epochs):
    for _ in range(len(X_train_scaled) // batch_size):
        noise = np.random.uniform(0, 1, size=(batch_size, X_train_scaled.shape[1]))
        synthetic_samples = generator.predict(noise)
        real_samples = X_train_scaled[np.random.randint(0, X_train_scaled.shape[0], batch_size)]
        
        combined_samples = np.concatenate([real_samples, synthetic_samples])
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        labels += 0.05 * np.random.random(labels.shape)  # Add noise to labels
        
        discriminator.trainable = True
        discriminator_loss = discriminator.train_on_batch(combined_samples, labels)
        
        noise = np.random.uniform(0, 1, size=(batch_size, X_train_scaled.shape[1]))
        misleading_labels = np.zeros((batch_size, 1))
        
        discriminator.trainable = False
        gan_loss = gan.train_on_batch(noise, misleading_labels)
    
    print(f'Epoch {epoch+1}, Discriminator Loss: {discriminator_loss[0]}, GAN Loss: {gan_loss}')
    if (epoch+1) % 10 == 0:
        generator.save_weights(f'generator_weights_epoch_{epoch+1}.h5')

# Generate synthetic samples using the trained generator
synthetic_samples = generator.predict(np.random.uniform(0, 1, size=(len(X_test_scaled), X_test_scaled.shape[1])))

# Combine the real and synthetic samples for the ensemble
ensemble_X_test = np.concatenate([X_test_scaled, synthetic_samples])
ensemble_y_test = np.concatenate([y_test] * 2)  # Duplicate labels for the combined dataset

# Train a classifier (e.g., MLP) on the ensemble dataset
classifier = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
classifier.fit(ensemble_X_test, ensemble_y_test)

# Make predictions using the trained classifier
ensemble_predictions = classifier.predict(X_test_scaled)

# Evaluate the ensemble model
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print(f'Ensemble accuracy: {ensemble_accuracy * 100:.2f}%')
