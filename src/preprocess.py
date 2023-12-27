import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Data:
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_fifth = MinMaxScaler(feature_range=(0, 1))
    # data = None
    
    # def __init__(self, data):
    #     self.data = data
        

    def preprocess_data(self, series):
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
    
    
    preprocess_and_normalize = lambda self, series: self.normalize(self.preprocess_data(series))
                
        
