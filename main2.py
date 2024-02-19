import random
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate data and ensure balanced representation
def generate_data(num_samples, seed=None):
    random.seed(seed)
    data = []
    labels = []
    for i in range(num_samples):
        num = random.randint(0, 2**11 - 1)  # Generate random numbers within bin(0) to bin(2047)
        binary_str = str(bin(num))[2:].zfill(11)  # Convert to binary string with fixed length
        data.append([int(digit) for digit in binary_str])
        labels.append(1 if num % 3 == 0 else 0)

    # Balance the dataset before splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=seed, stratify=labels)
    if sum(y_train) / len(y_train) < 0.33:  # Check train set balance
        # Oversample or undersample as needed using appropriate techniques like SMOTE
        raise ValueError("Train set imbalance exceeds threshold. Apply balancing techniques.")
    return X_train, X_test, y_train, y_test

# Define MLP structure using PyTorch
class MLP(nn.Module):
    def __init__(self, num_inputs, num_hidden_units, num_outputs, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, num_hidden_units))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(num_hidden_units, num_hidden_units))
        self.layers.append(nn.Linear(num_hidden_units, num_outputs))

    def forward(self, inputs):
        activations = inputs
        for layer in self.layers:
            z = layer(activations)
            if layer != self.layers[-1]:  # Apply ReLU activation except for the output layer
                activations = torch.relu(z)
        return activations  # Output of the final layer

# Define training and evaluation functions
def train(model, optimizer, criterion, X_train, y_train, epochs=100, batch_size=32, verbose=0):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        for batch in range(0, len(X_train), batch_size):
            X_batch = torch.tensor(X_train[batch:batch+batch_size], dtype=torch.float32)
            y_batch = torch.tensor(y_train[batch:batch+batch_size], dtype=torch.float32)
            # Forward pass and calculate loss
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and batch % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch}/{len(X_train)//batch_size}, Loss: {loss.item():.4f}")

def evaluate(model, X_test, y_test):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        predictions = model(torch.tensor(X_test, dtype=torch.float32))

    # Calculate accuracy
    correct_predictions = (predictions.round() == torch.tensor(y_test)).sum().item()
    accuracy = correct_predictions / len(y_test) * 100
    print(f"Accuracy on test set: {accuracy:.2f}%")

# Hyperparameters (adjust as needed)
num_inputs = 11
num_hidden_units = 32
num_outputs = 1
num_layers = 4
learning_rate = 0.01
epochs = 100
batch_size = 32

# Generate and
# Generate and preprocess data (consider normalization using StandardScaler)
X_train, X_test, y_train, y_test = generate_data(10000, seed=42)  # Increase training data size
scaler = StandardScaler()  # Optional: Normalize data for better convergence
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create MLP model and optimizer
model = MLP(num_inputs, num_hidden_units, num_outputs, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train(model, optimizer, criterion, X_train_scaled, y_train, epochs, batch_size)

# Evaluate the model on the test set
evaluate(model, X_test_scaled, y_test)

# Optional: Save the trained model for future use
torch.save(model.state_dict(), "mlp_model.pt")