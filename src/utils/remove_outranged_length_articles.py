import pandas as pd



#Function to remove articles which less than minimum size i.e:, meaning less and also oversized articles.

def articles_by_length(data: pd.DataFrame ,min_length: int, max_length: int)-> pd.DataFrame:
    initial_data_length = data.shape[0]

    #Removing articles below minimum length
    data = data[data['word_count'] >= min_length]
    print(f"removed {initial_data_length - data.shape[0]} articles which are less than {min_length} words")

    #Removing articles below minimum length
    initial_data_length = data.shape[0]
    data = data[data['word_count'] <= max_length]
    print(f"removed {initial_data_length - data.shape[0]} articles which are greater than {max_length} words")

    return data