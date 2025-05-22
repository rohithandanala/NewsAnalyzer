import pandas as pd

#This function removes duplicate data
def remove_duplicate_data(data:pd.DataFrame, subset: str) -> pd.DataFrame:
    initial_size = data.shape[0]
    data = data.drop_duplicates(subset=[subset], keep='first')
    print(f"Removed {initial_size - data.shape[0]} duplicate rows from data")
    return data