import pandas as pd

def SaveData(data: pd.DataFrame, path):
    data.to_csv(path)
    print(f'saved processed data at {path}')