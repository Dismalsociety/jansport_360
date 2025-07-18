# jansport_360
Neural Network


Structure:
Imports: Brings in tools needed to build and train neural networks, read data files, and prepare the data.
torch: Core library for deep learning.
pandas: Reads .csv or .xlsx files easily.
Dataset & DataLoader: Let us prepare data for training in batches.
MinMaxScaler: Normalizes numbers to a 0–1 range (this helps neural networks learn better).
train_test_split: Divides the dataset into training and testing sets.
Parameters: Sets values that control how the models train.
SEQ_LENGTH: How many flow readings to look at before predicting pressure.
BATCH_SIZE: How many samples we feed into the model at once.
EPOCHS: How many times we repeat training on the dataset.
LR: How fast the model updates its internal weights.
HIDDEN_SIZE: Controls how many "neurons" the model uses internally.
Load Data: Reads the excel file and returns just the columns we care about: flow and pressure.
Dataset Class: Prepares the data in "sliding windows" where:
Input = 10 flow rate values
Output = pressure at the next time step
This class breaks the time series into manageable training samples
__len__() tells PyTorch how many samples there are.
__getitem__() returns one (input, label) pair at a time for training.
Model Definitions: Build the three different neural networks
FCNN (Fully Connected Neural Network)
Treats the flow sequence like a flat vector and runs it through two dense layers.
LSTM (Long Short-Term Memory)
Uses a memory-based model that understands sequences better than FCNN.
GRU (Gated Recurrent Unit)
Similar to LSTM but faster and simpler.
Sometimes GRU works just as well as LSTM and is more efficient.
Training Loop: Feeds training data into the model and adjusts the model’s weights to reduce the prediction error.
It's reusable: works for any of the three models.
It loops through data in batches, makes predictions, calculates how wrong the predictions are (loss), updates model weights to get better.
Main Program: Put together and run it 
This is the central function that loads and normalizes data, prepares training sequences, initializes the models, trains each model using the function above
Keep your code organized and modular.
You can change models or data without touching training logic.
