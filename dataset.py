import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"The file {self.data_path} does not exist.")
        
        data = pd.read_csv(self.data_path)
        return data

    def split_data(self, data, test_size=0.2, random_state=42):
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        return train_data, test_data