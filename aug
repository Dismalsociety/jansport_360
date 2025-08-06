import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import math

# Parameters
sequence_length = 10    
predict_length = 5     
batch_size = 32     
learning_rate = 0.001 
epochs = 100
dropout_rate = 0.3  # ADD: Dropout for regularization
weight_decay = 1e-4  # ADD: L2 regularization
noise_std = 0.01  # ADD: Noise standard deviation for data augmentation

# Use GPU instead of CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
print("Loading data...")
df = pd.read_excel('generated_training_data_2.xlsx')
flow = df['FLow Rate'].values 
pressure = df['Pressure'].values 

# Combine features
data = np.column_stack([flow, pressure])

# IMPROVED: Split data BEFORE creating sequences to prevent leakage
total_samples = len(data)
train_size = int(0.7 * total_samples)  # 70% train
val_size = int(0.15 * total_samples)   # 15% validation
test_size = total_samples - train_size - val_size  # 15% test

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Normalize using ONLY training data statistics
scaler = StandardScaler()
train_data_normalized = scaler.fit_transform(train_data)
val_data_normalized = scaler.transform(val_data)  # Use train statistics
test_data_normalized = scaler.transform(test_data)  # Use train statistics

# Create sequences for each split
def create_sequences(data_normalized, seq_len, pred_len):
    X_data = []
    Y_data = []
    for i in range(len(data_normalized) - seq_len - pred_len + 1):
        input_seq = data_normalized[i:i+seq_len]
        X_data.append(input_seq)
        target = data_normalized[i+seq_len:i+seq_len+pred_len, 1]
        Y_data.append(target)
    return np.array(X_data), np.array(Y_data)

# Create sequences for each set
X_train, Y_train = create_sequences(train_data_normalized, sequence_length, predict_length)
X_val, Y_val = create_sequences(val_data_normalized, sequence_length, predict_length)
X_test, Y_test = create_sequences(test_data_normalized, sequence_length, predict_length)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
Y_train = torch.FloatTensor(Y_train)
X_val = torch.FloatTensor(X_val)
Y_val = torch.FloatTensor(Y_val)
X_test = torch.FloatTensor(X_test)
Y_test = torch.FloatTensor(Y_test)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# SIMPLIFIED Model with Regularization
class RegularizedModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(RegularizedModel, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(2, 64)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Single LSTM (simpler than bidirectional)
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=64,  # Reduced from 128
            num_layers=2,    # Added depth instead of bidirectional
            dropout=dropout_rate if dropout_rate > 0 else 0,  # Dropout between LSTM layers
            batch_first=True
        )
        
        # Output layers with dropout
        self.fc1 = nn.Linear(64, 32)
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(32, predict_length)
        self.relu = nn.ReLU()
        
        # Batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(64)

    def forward(self, x):
        # Input processing
        x = self.input_layer(x)
        x = self.input_dropout(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state (from last layer)
        last_hidden = hidden[-1]  # Shape: (batch, hidden_size)
        
        # Apply batch norm
        x = self.batch_norm(last_hidden)
        
        # Output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        
        return x

# Initialize model with regularization
model = RegularizedModel(dropout_rate=dropout_rate).to(device)

# Loss function and optimizer with weight decay
MSE = nn.MSELoss()
Adam = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler with early stopping patience
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    Adam, 
    mode='min',
    factor=0.5,
    patience=10,  # Increased patience
    verbose=True,
    min_lr=1e-6
)

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=15)

# Training with early stopping
print("\nTraining with regularization and data augmentation...")
train_losses = []
val_losses = []
learning_rates = []
best_model_state = None
best_val_loss = float('inf')

# Track augmentation effect
augmented_train_losses = []  # Loss with augmented data
clean_train_losses = []      # Loss without augmentation for comparison

for epoch in range(epochs):
    # Training phase
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Shuffle training data each epoch
    indices = torch.randperm(len(X_train))
    X_train_shuffled = X_train[indices]
    Y_train_shuffled = Y_train[indices]
    
    for i in range(0, len(X_train_shuffled), batch_size):
        batch_x = X_train_shuffled[i:i+batch_size].to(device)
        batch_y = Y_train_shuffled[i:i+batch_size].to(device)
        
        # DATA AUGMENTATION: Add Gaussian noise to inputs during training
        # This helps prevent overfitting by making the model robust to small variations
        if model.training:
            noise = torch.randn_like(batch_x) * noise_std
            batch_x_augmented = batch_x + noise
        else:
            batch_x_augmented = batch_x
        
        outputs = model(batch_x_augmented)
        loss = MSE(outputs, batch_y)
        
        Adam.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        Adam.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_train_loss = total_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # Optional: Evaluate on clean training data to see augmentation effect
    model.eval()
    with torch.no_grad():
        clean_loss = 0
        clean_batches = 0
        for i in range(0, min(len(X_train), 320), batch_size):  # Sample for speed
            batch_x = X_train[i:i+batch_size].to(device)
            batch_y = Y_train[i:i+batch_size].to(device)
            outputs = model(batch_x)
            clean_loss += MSE(outputs, batch_y).item()
            clean_batches += 1
        clean_train_losses.append(clean_loss / clean_batches if clean_batches > 0 else 0)
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.to(device))
        val_loss = MSE(val_outputs, Y_val.to(device))
        val_losses.append(val_loss.item())
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = Adam.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    # Early stopping check
    early_stopping(val_loss)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
    
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

# Testing
print("\nTesting...")
model.eval()

all_predictions = []
all_actuals = []

with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        batch_x = X_test[i:i+batch_size].to(device)
        batch_y = Y_test[i:i+batch_size]
        
        predictions = model(batch_x)
        
        all_predictions.append(predictions.cpu().numpy())
        all_actuals.append(batch_y.numpy())

# Process results
predictions = np.vstack(all_predictions) if all_predictions else np.array([])
actuals = np.vstack(all_actuals) if all_actuals else np.array([])

# Convert back to original scale
if len(predictions) > 0:
    pressure_mean = scaler.mean_[1]
    pressure_std = scaler.scale_[1]
    
    predictions_real = predictions * pressure_std + pressure_mean
    actuals_real = actuals * pressure_std + pressure_mean
    
    # Calculate metrics
    correlation, p_value = pearsonr(predictions_real.flatten(), actuals_real.flatten())
    ss_res = np.sum((actuals_real - predictions_real) ** 2)
    ss_tot = np.sum((actuals_real - np.mean(actuals_real)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\nTest Set Performance:")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  R-squared: {r_squared:.4f}")
    
    # Calculate and print generalization gap
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    gap = final_val_loss - final_train_loss
    print(f"\nGeneralization Gap: {gap:.4f}")
    print(f"  Final Train Loss: {final_train_loss:.4f}")
    print(f"  Final Val Loss: {final_val_loss:.4f}")

# Plotting
plt.figure(figsize=(20, 5))

# Plot 1: Training and validation loss
plt.subplot(1, 4, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Val Loss', alpha=0.8)
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Learning rate schedule
plt.subplot(1, 4, 2)
plt.plot(learning_rates)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Plot 3: Predictions vs Actual (if test data exists)
if len(predictions) > 0:
    plt.subplot(1, 4, 3)
    sample_size = min(100, len(actuals_real))
    plt.plot(actuals_real[:sample_size, 0], label='Actual', alpha=0.7)
    plt.plot(predictions_real[:sample_size, 0], label='Predicted', alpha=0.7)
    plt.title('Test Set: Predictions vs Actual')
    plt.xlabel('Test Sample')
    plt.ylabel('Pressure (psiA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot for correlation
    plt.subplot(1, 4, 4)
    plt.scatter(actuals_real.flatten(), predictions_real.flatten(), alpha=0.5, s=1)
    plt.plot([actuals_real.min(), actuals_real.max()], 
             [actuals_real.min(), actuals_real.max()], 
             'r--', label='Perfect Prediction')
    plt.title(f'Actual vs Predicted (R²={r_squared:.3f})')
    plt.xlabel('Actual Pressure')
    plt.ylabel('Predicted Pressure')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nDone!")
