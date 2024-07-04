from get_data import CIFAR10_Dataset
import CNN
from torch.utils.data import DataLoader
import torch.optim as opt
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch
import matplotlib.pyplot as plt 
import os
from Mytest import test_func

def train_func(path: str, lr: str, batch_size: int, num_epochs: int, dropout_p: float, decay_rate: float, block_layer_num: int, train_or_test: str):
    if train_or_test == "test":
        return ["测试模式", "测试模式", "测试模式", test_func(f"..\check_points\{path}",block_layer_num)[0] , None]
    dropout_p = float(dropout_p)
    resnet18_model=CNN.ResNet([block_layer_num]*4, 10, dropout_p)
    if path != "":
        checkpoint = torch.load(f"..\check_points\{path}")
        resnet18_model.load_state_dict(checkpoint['model_state_dict']) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    """
    lr = 0.000004
    batch_size = 256
    num_epochs = 100  # 设置训练轮数
    """
    lr = float(lr)
    
    train_data = CIFAR10_Dataset("train")
    eval_data = CIFAR10_Dataset("eval")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size)
    
    model = resnet18_model
    model.to(device)
    #optimizer = opt.SGD(model.parameters(), lr=lr)
    optimizer = opt.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=decay_rate)  # 每1个epoch学习率乘以0.1
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    #criterion = nn.SmoothL1Loss()
    criterion = criterion.to(device)
    
    max_acc = 0
    counter = 0
    filename = "../check_points/checkpoint-0.pth"
    if os.path.exists(filename):
        while os.path.exists(f"../check_points/checkpoint-{counter}.pth"):
            counter += 1 
        filename = f"../check_points/checkpoint-{counter}.pth"
    x, y = [], [] 
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for i,(inputs,labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移至设备
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播
            #loss = criterion(outputs,torch.nn.functional.one_hot(labels.long().cuda(), num_classes=10))  # 计算损失
            loss = criterion(outputs, labels.long().cuda())
            loss.backward()  # 反向传播:
            optimizer.step()  # 更新模型参数
            running_loss += loss.item()  # 累加损失
        scheduler.step()
        model.eval()
        acc = 0
        for j,(inputs,labels) in enumerate(eval_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移至设备
            outputs = model(inputs)
            for k in range(len(outputs)) :
                s = torch.argmax(outputs[k])
                if s == labels[k] :
                    acc +=1
        checkpoint = {  
            'learning_rate': lr,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'model_state_dict': model.state_dict(),  
            'optimizer_state_dict': optimizer.state_dict(),   
            'loss': running_loss,  
        }
        x.append(epoch)
        y.append(running_loss)
        
        if acc/len(eval_data.data) > max_acc:
            torch.save(checkpoint, filename)
            max_acc = acc/len(eval_data.data)
            choosen_epoch = epoch
            
        print(acc/len(eval_data.data),running_loss)
        print(epoch)
        
    plt.ylim(min(y), max(y)) 
    plt.plot(x, y, color="red")
    plt.savefig(f"../loss/runing-loss-{counter}.png")
    
    return [max_acc, choosen_epoch, f"../check_points/checkpoint-{counter}.pth",test_func(f"..\check_points\checkpoint-{counter}.pth",block_layer_num)[0] , f"../loss/runing-loss-{counter}.png"]
    