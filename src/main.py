from data import Data, get_data
from config import API_KEY


def main():
    
    # give 1 min intraday context
    # should give 30 min intraday context too?
    # could organize 1 min data into 30 min past a certain time
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=1min&outputsize=full&apikey=${API_KEY}'
    url_30 = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=30min&outputsize=full&apikey=${API_KEY}'

    # saves scalers to class
    data = Data()
    normalized_features = data.preprocess(get_data(url)["Time Series (1min)"])
    normalized_30_features = data.preprocess(get_data(url_30)["Time Series (30min)"])
    
    print(len(normalized_features) / len(normalized_30_features))

        
    print(data.generate_30_min_data(normalized_features))
    print(normalized_30_features)
    
    
    # labels = data.compute_scores(normalized_features[:, 3])
    
    for i in data.create_train(normalized_features):
        print(i)
    
    # print(labels)
    
    
    
    
    
    
    

if "__main__" == __name__:
    main()