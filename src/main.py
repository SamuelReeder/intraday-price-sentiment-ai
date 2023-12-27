from preprocess import Data
from data import get_data
from config import API_KEY


def main():
    
    # give 1 min intraday context
    # should give 30 min intraday context too?
    # could organize 1 min data into 30 min past a certain time
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=1min&outputsize=full&apikey=${API_KEY}'

    # saves scalers to class
    data = Data()
    normalized_features = data.preprocess_and_normalize(get_data(url)["Time Series (1min)"])
    print(normalized_features)
    
    labels = data.compute_scores(normalized_features[:, 3])
    
    
    print(labels)
    
    
    
    
    
    
    

if "__main__" == __name__:
    main()