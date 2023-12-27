from preprocess import Data
from data import get_data
from secrets import API_KEY


def main():
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=1min&outputsize=full&apikey=${API_KEY}'

    # saves scalers to class
    data = Data();
    normalized_features = data.preprocess_and_normalize(get_data(url))
    print(normalized_features)
    
    
    
    
    
    

if "__main__" == __name__:
    print("Hello World!")