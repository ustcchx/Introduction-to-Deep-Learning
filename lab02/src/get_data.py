import pickle
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

class NoSuchNameError(Exception):  
    pass 

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_CIFAR10_data_batch(batch_id = 1) :
    if batch_id != 6:
        file_dir = '../data/data_batch_' + str(batch_id)
    else:
        file_dir = '../data/test_batch'
    dict_ = unpickle(file_dir)
    img = dict_[b'data']
    labels = dict_[b'labels']
    return np.array(img),np.array(labels)

class CIFAR10_Dataset(Dataset) :
    
    def __init__(self,mode = "train"):
        if mode == "train":
            self.data ,self.target = load_CIFAR10_data_batch(1)
            for i in range(2,5) :
                temp_data ,temp_target = load_CIFAR10_data_batch(i)
                self.data = np.concatenate([self.data,temp_data])
                self.target = np.concatenate([self.target,temp_target])
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.49139968 ,0.48215841, 0.44653091), 
                    std=(0.20220212, 0.19931542, 0.20086346)
                )
            ])
        else:
            if mode == "test":
                data ,target = load_CIFAR10_data_batch(6)
            elif mode == "eval":
                data ,target = load_CIFAR10_data_batch(5)
            else:
                raise NoSuchNameError("Only 'train', 'eval', 'test'")
            self.data = data
            self.target = target
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.49139968 ,0.48215841, 0.44653091), 
                    std=(0.20220212, 0.19931542, 0.20086346)
                )
            ])
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
      return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.target[idx]
        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)
        return img,label