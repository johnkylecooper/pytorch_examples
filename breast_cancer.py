# %% [markdown]
# Breast Cancer Prediction Using Neural Networks in PyTorch

# %% [markdown]
# Requirements

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# Device Configuration

# %%
# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# Data Collection and Preprocessing

# %%
# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# %%
# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Standardize the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# Convert data to PyTorch tensors and move it to GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# %% [markdown]
# Neural Network Architecture

# %%
# Define the neural network architecture
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# %%
# Define hyperparameters
input_size = X_train.shape[1]
hidden_size = 64  # neurons
output_size = 1  # single neuron
learning_rate = 0.001
num_epochs = 100

# %%
# Initialize the neural network and move it to the GPU
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# %%
# Define the loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %% [markdown]
# Training the Neural Network

# %%
# Training the model
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1, 1))
    loss.backward()
    optimizer.step()

    # calculate accuracy
    with torch.no_grad():
        predicted = outputs.round()
        correct = (predicted == y_train.view(-1, 1)).float().sum()
        accuracy = correct / y_train.size(0)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1} / {num_epochs}], Loss : {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%")

# %% [markdown]
# Model Evaluation

# %%
# Evaluation on training set
model.eval()
with torch.no_grad():
    outputs = model(X_train)
    predicted = outputs.round()
    correct = (predicted == y_train.view(-1, 1)).float().sum()
    accuracy = correct / y_train.size(0)
    print(f"Accuracy on training data: {accuracy.item() * 100:.2f}%")

# %%
# Evaluation on testing set
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted = outputs.round()
    correct = (predicted == y_test.view(-1, 1)).float().sum()
    accuracy = correct / y_test.size(0)
    print(f"Accuracy on testing data: {accuracy.item() * 100:.2f}%")


