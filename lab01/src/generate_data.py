import numpy as np
import random
import pandas as pd
import argparse 
parser = argparse.ArgumentParser()  
parser.add_argument('--N', type=int, default=10000 , help='输入均匀采样样本个数')  
args = parser.parse_args()

def main():
    N = args.N 
    
    x = [random.uniform(1,16) for _ in range(N)]
    y = [np.log(x)/np.log(2) + np.cos(x*np.pi/2) for x in x]
    dataset = np.array([[x[i], y[i]] for i in range(N)])
    
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)  
    val_size = int(0.1 * total_samples)  
    
    # 打乱数据集的索引  
    indices = list(range(total_samples))  
    random.shuffle(indices)  
     
    # 分割索引  
    train_indices = indices[:train_size]  
    val_indices = indices[train_size:train_size+val_size]  
    test_indices = indices[train_size+val_size:]  
      
    # 创建子数据集  
    train_set = dataset[train_indices]
    val_set = dataset[val_indices]
    test_set = dataset[test_indices]
    
    pd_train = pd.DataFrame(train_set)
    pd_val = pd.DataFrame(val_set)
    pd_test = pd.DataFrame(test_set)
    
    pd_train.to_csv(f'../dataset/train{N}.csv', index=False)
    pd_val.to_csv(f'../dataset/val{N}.csv', index=False)
    pd_test.to_csv(f'../dataset/test{N}.csv', index=False)
    
    print("Generate dataset successfully")
    
if __name__ == '__main__':  
    main()