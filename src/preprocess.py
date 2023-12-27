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
    
    
    preprocess_and_normalize = lambda self, series: self.normalize(self.preprocess_data(series))
                
        
