import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

file = "exp"
data = pd.read_csv(f'data/{file}.csv')

x = data['x'].values.reshape(-1, 1)
y = data['y'].values.reshape(-1, 1)

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)

class SigmoidNN(nn.Module):
    def __init__(self, hidden_layer_sz):
        super(SigmoidNN, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(1, hidden_layer_sz),
            nn.Sigmoid(),
            nn.Linear(hidden_layer_sz, 1)
        )
    
    def forward(self, x):
        return self.stack(x)

def train_and_plot(hidden_layer_sz, epochs=1000, lr=0.01):
    model = SigmoidNN(hidden_layer_sz)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    
    # Train the model
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(x_train)
        loss = criterion(pred, y_train)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)

    final_loss = criterion(y_test_pred, y_test).item()

    # Plot the results
    y_test_pred_np = y_test_pred.numpy()
    y_test_np = y_test.numpy()
    x_test_np = x_test.numpy()

    plt.scatter(x_test_np, y_test_np, color='blue', label='Actual Data')
    plt.scatter(x_test_np, y_test_pred_np, color='red', label='Predicted Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Hidden Layer Size: {hidden_layer_sz}, Final Loss: {final_loss:.2f}')
    plt.savefig(f'sigmoid_plots/{file}/plot_hidden_{hidden_layer_sz}_loss_{final_loss:.2f}.png')
    plt.close()

if file == "big_poly":
    hidden_layer_szs = [2000, 4000, 16000, 48000]
else:
    hidden_layer_szs = [2, 20, 200, 2000]

shutil.rmtree(f"sigmoid_plots/{file}")
os.makedirs(f"sigmoid_plots/{file}")

for sz in hidden_layer_szs:
    train_and_plot(hidden_layer_sz=sz)