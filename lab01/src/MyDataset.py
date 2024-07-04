import torch.utils.data as data
import pandas as pd
import numpy as np
import torch
from torch import tensor 

class Mydataset(data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index] 
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)

def get_data_loader(file_name: str, batch_size: int, shuffle=False):
    df = pd.read_csv(file_name)
    data_list = np.array(df)
    x_tensor_list = []
    y_tensor_list = []
    for item in data_list:
        x_tensor_list.append(tensor([item[0]], dtype=torch.float))
        y_tensor_list.append(tensor([item[1]], dtype=torch.float))
    dataset = Mydataset(x_tensor_list, y_tensor_list)
    return data.DataLoader(dataset, batch_size=batch_size)
