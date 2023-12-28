import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests
from typing import Dict, Any


def get_data(url: str) -> Dict[str, Any]:
    r = requests.get(url)
    data = r.json()
    return data

class Data:
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_fifth = MinMaxScaler(feature_range=(0, 1))
    # data = None
    
    # def __init__(self, data):
    #     self.data = data
        

    def preprocess(self, series):
        arr = [list(float(series[i][j]) for j in series[i]) for i in series]

        return np.array(arr, dtype=np.float32)
    
    
    def normalize(self, features):
        first_four_features = features[:, :4]
        flattened_features = first_four_features.flatten().reshape(-1, 1)
        self.scaler.fit(flattened_features)

        normalized_first_four = self.scaler.transform(first_four_features.reshape(-1, 1)).reshape(first_four_features.shape)

        fifth_feature = features[:, 4].reshape(-1, 1)
        normalized_fifth = self.scaler_fifth.fit_transform(fifth_feature)

        normalized_features = np.hstack([normalized_first_four, normalized_fifth])

        return normalized_features
    
    

    def denormalize(self, normalized_features):
        inverse_first_four = self.scaler.inverse_transform(normalized_first_four.reshape(-1, 1)).reshape(first_four_features.shape)

        inverse_fifth = self.scaler_fifth.inverse_transform(normalized_fifth)

        inverted_features = np.hstack([inverse_first_four, inverse_fifth])

        return inverted_features
    
    def compute_scores(self, prices: np.ndarray, look_ahead: int = 30, decay_factor: float = 0.05, moving_average_period: int = 10) -> np.ndarray:

        # moving average
        smoothed_prices = np.convolve(prices, np.ones(moving_average_period)/moving_average_period, mode='valid')

        scores = np.zeros(len(smoothed_prices) - look_ahead)

        for i in range(len(scores)):
            # Calculate the moving average price changes for the look-ahead period
            future_prices = smoothed_prices[i + 1:i + 1 + look_ahead]
            price_changes = future_prices - smoothed_prices[i]

            # Apply time decay
            time_decay = np.exp(-decay_factor * np.arange(look_ahead))

            # Combine time decay and magnitude of changes
            weights = time_decay * np.abs(price_changes)

            # Calculate the weighted sum of price changes
            scores[i] = np.sum(weights * price_changes)

        # Normalize the scores to a 0-10 scale
        min_score, max_score = np.min(scores), np.max(scores)
        normalized_scores = (scores - min_score) / (max_score - min_score) * 10
        
        return normalized_scores
    
    
    def generate_30_min_data(self, data) -> np.ndarray:
        
        thirty = []
        for i in range(0, len(data) // 30):
            j = i * 30
            high = max(data[j:j+30,1])
            low = min(data[j:j+30,2])
            open_value = data[j][0]
            close = data[j+29][3]
            volume = sum(data[j:j+30][4])
            thirty.append([open_value, high, low, close, volume])
            
        return np.array(thirty, dtype=np.float32)
    
    
    def generate_training_example(self, data) -> np.ndarray:
        min_length = 64
        thirty_len = 32
        
        start = 32 * 30
        
        arr = []
        for i in range(start, len(data)):
            thirty = self.generate_30_min_data(data[i - start:i])
            min_data = data[i - min_length:i]
            arr.append(thirty + min_data)
            
    def create_train(self, data):
        for i in range(32 * 30 + 64, len(data)):
            thirty = self.generate_30_min_data(data[i - 32 * 30 - 64:i - 64])
            min_data = data[i - 64:i]
            yield np.concatenate((thirty, min_data))
    
    
    preprocess_and_normalize = lambda self, series: self.normalize(self.preprocess_data(series))
                
        
