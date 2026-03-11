# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset

Problem Statement:

Stock market prices change frequently due to various factors such as economic conditions, company performance, and market trends. Predicting future stock prices is challenging because the data is sequential and time-dependent. Traditional machine learning models often fail to capture these temporal patterns effectively.

Therefore, there is a need to develop a model that can learn from historical stock price data and identify patterns over time. A Recurrent Neural Network (RNN) is suitable for this task because it is designed to process sequential data and remember past information. The problem is to build an RNN model that uses historical closing price data to predict future stock prices with better accuracy.

<img width="891" height="684" alt="image" src="https://github.com/user-attachments/assets/aa398994-3cb2-4404-83bc-ce4ec55ca89a" />
<img width="786" height="687" alt="image" src="https://github.com/user-attachments/assets/d50119f6-d6fc-41cf-948a-7d53d51c91f1" />



## DESIGN STEPS

### STEP 1: 
Load and normalize data, create sequences.

### STEP 2: 

Convert data to tensors and set up DataLoader.

### STEP 3: 

Define the RNN model architecture

### STEP 4: 

Summarize, compile with loss and optimizer.

### STEP 5: 
Train the model with loss tracking.


### STEP 6: 

Predict on test data, plot actual vs. predicted prices.



## PROGRAM

### Name: ABISHA RANI S

### Register Number: 212224040012

```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

## Step 1: Load and Preprocess Data
# Load training and test datasets
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

df_train.head()

# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)


x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

## Step 2: Define RNN Model
class RNNModel(nn.Module):
  def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel,self).__init__()
    self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc=nn.Linear(hidden_size,output_size)
  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

!pip install torchinfo

from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## Step 3: Train the Model
def train_model(model,train_loader,criterion,optimizer,epochs=20):
  train_losses=[]
  model.train() # Corrected typo: model.tarin() to model.train()
  for epoch in range(epochs):
    total_loss=0
    for x_batch,y_batch in train_loader:
      x_batch=x_batch.to(device)
      y_batch=y_batch.to(device)
      optimizer.zero_grad()
      output=model(x_batch)
      loss=criterion(output,y_batch)
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    train_losses.append(total_loss/len(train_loader))
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
  return train_losses # Moved return statement outside the loop

# Call the train_model function and store its result
training_losses_history = train_model(model, train_loader, criterion, optimizer, epochs=20)

print('Name: ABISHA RANI S')
print('Register Number:212224040012')
plt.plot(training_losses_history, label='Training Loss') # Used the returned variable
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name: ABISHA RANI S')
print('Register Number: 212224040012')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')



```

### OUTPUT

<img width="1375" height="202" alt="image" src="https://github.com/user-attachments/assets/f2e7378c-b9d6-4db1-a210-894cad0a3f13" />

<img width="1115" height="317" alt="image" src="https://github.com/user-attachments/assets/6e192879-f86d-43ff-a515-9c94025ad075" />


## Training Loss Over Epochs Plot

<img width="720" height="708" alt="image" src="https://github.com/user-attachments/assets/970272e0-c4e9-49f5-8ab8-41e57ad196d4" />


## True Stock Price, Predicted Stock Price vs time

<img width="1205" height="661" alt="image" src="https://github.com/user-attachments/assets/3c2eb25d-7350-432c-8f59-43c4c774f684" />


### Predictions

<img width="471" height="66" alt="image" src="https://github.com/user-attachments/assets/a9361d16-9bc8-4121-a6e0-f845b233de5b" />


## RESULT

Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
