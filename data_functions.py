import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)   
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

def get_train_data(data, window_size):
    """Get daily training data. Window_size = number of previous days used to predict next days price"""
    
    X, y = [], []
    
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    
    return np.array(X), np.array(y)
    
def get_test_data(data, testing_data, scaler, window_size):
    """Get daily testing data."""
    
    raw = data['Price'][len(data) - len(testing_data) - window_size:].values
    raw = raw.reshape(-1, 1)
    raw = scaler.transform(raw)
    
    X, y = [], []
    
    for i in range(window_size, raw.shape[0]):
        X.append(raw[i-window_size:i, 0])
        y.append(raw[i])
        
    X = np.array(X)
    y = np.array(y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y
    
def get_weekly_train_data(data, window_size):
    
    X, y = [], []
    
    for i in range(window_size, len(data), 5):
        X.append(data[i-window_size:i])
        y.append(data[i:i+5])
        
    X = np.array(X)
    y = np.array(y)
    
    y = np.reshape(y, (y.shape[0], y.shape[1]))    
    
    return X, y
    
def get_weekly_test_data(data, testing_data, scaler, window_size):
    raw = data['Price'][len(data) - len(testing_data) - window_size:].values
    raw = raw.reshape(-1, 1)
    raw = scaler.transform(raw)
    
    X, y = [], []
    
    for i in range(window_size, raw.shape[0], 5):
        X.append(raw[i-window_size:i, 0])
        y.append(raw[i:i+5])
        
    X = np.array(X)
    y = np.array(y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1]))
    
    return X, y
    
def get_monthly_train_data(data, window_size):
    
    X, y = [], []
    
    for i in range(window_size, len(data), 20):
        X.append(data[i-window_size:i])
        y.append(data[i:i+20])
        
    X = np.array(X)
    y = np.array(y)
    
    y = np.reshape(y, (y.shape[0], y.shape[1]))    
    
    return X, y
    
def get_monthly_test_data(data, testing_data, scaler, window_size):
    raw = data['Price'][len(data) - len(testing_data) - window_size:].values
    raw = raw.reshape(-1, 1)
    raw = scaler.transform(raw)
    
    X, y = [], []
    
    for i in range(window_size, raw.shape[0], 20):
        X.append(raw[i-window_size:i, 0])
        y.append(raw[i:i+20])
        
    X = np.array(X)
    y = np.array(y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1]))
    
    return X, y

def sma(data, window):
    """ Calculate the simple moving average"""
    
    sma = data.rolling(window=window).mean()
    
    return sma
    
def bollinger(data, sma, window):
    """ Calculate upper and lower bollinger bands"""
    
    std = data.rolling(window=window).std()
    upper_bb = sma + std*2
    lower_bb = sma - std*2
    
    return upper_bb, lower_bb