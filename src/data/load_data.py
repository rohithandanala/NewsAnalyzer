import pandas as pd


#Read and return data as pandas dataframe
def fetch_data(path: str) -> pd.DataFrame:
    print(f'Loading data from {path}')
    data = pd.read_csv(path)
    return data