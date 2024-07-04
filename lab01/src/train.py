import argparse  
from torch.optim.lr_scheduler import StepLR
from process_bar import progress_bar
from ParamAction_class import ParamAction
import MyDataset
from MLP_model import MLPs
import torch.optim as opt
import torch.nn as nn
import torch
import matplotlib.pyplot as plt 
import os

parser = argparse.ArgumentParser()  
parser.add_argument('--lr', type=float, default=3e-3,action=ParamAction , help='输入学习率, default:3e-3')
parser.add_argument('--batch_size', type=int, default=256,action=ParamAction , help='输入batch_size, default:256')
parser.add_argument('--size_list', type=int, nargs='+',action=ParamAction , default=[1 ,512, 512, 512, 512, 1], help='输入每层传递值的size，default:[1 ,512, 512, 512, 512, 1]，注意开头收尾必须为1')
parser.add_argument('--num_epochs', type=int, default=10000,action=ParamAction , help='循环epoch，default:10000')
parser.add_argument('--activate', type=str, default='ReLU',action=ParamAction , help="激活函数类型: support ReLU/LeakyReLU/Tanh, default:ReLU")
parser.add_argument('--N', type=int, default=10000, help='对应N个样本，default:10000')
parser.add_argument('--load_parameters', type=str, help="选择最佳参数的checkpoint的路径，如果设置为checkpoint文件的参数，其他参数传入会覆盖checkpoint的参数")
parser.add_argument('--load_model', type=bool, default=1, help='1：载入checkpoint model自身参数，0：不载入checkpoint model自身参数，需要重新训练')

args = parser.parse_args()
actions = [a for a in parser._actions if isinstance(a, ParamAction)] 
        
def main():
    path = args.load_parameters
    load_model = args.load_model
    N = args.N
    
    # 赋给初值
    lr = args.lr
    batch_size = args.batch_size
    size_list = args.size_list
    num_epochs = args.num_epochs
    activate = args.activate
    
    if path :
        checkpoint = torch.load(path)
        lr = checkpoint["learning_rate"]
        batch_size = checkpoint["batch_size"]
        num_epochs = checkpoint["num_epochs"]
        size_list = checkpoint["size_list"]
        activate = checkpoint["activate"]
        print('如果设置为checkpoint文件的参数，其他参数传入会覆盖checkpoint的参数')

    if actions[0].was_provided:
        lr = args.lr
    if actions[1].was_provided:
        batch_size = args.batch_size
    if actions[3].was_provided:
        num_epochs = args.num_epochs
    if actions[2].was_provided:
        size_list = args.size_list
    if actions[4].was_provided:
        activate = args.activate
        
    print(f"num_epochs: {num_epochs}")
    print(f"batch size: {batch_size}")
    print(f"learning rate: {lr}")
    print(f"model size list: {size_list}")
    print(f"Activation Function: {activate}")
    
    train_iter = MyDataset.get_data_loader(f"../dataset/train{N}.csv", batch_size)
    test_iter = MyDataset.get_data_loader(f"../dataset/test{N}.csv", batch_size)
    val_iter = MyDataset.get_data_loader(f"../dataset/val{N}.csv", batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = MLPs(size_list, activate)
    
    if path and load_model:
        model.load_state_dict(checkpoint['model_state_dict']) 
    model.to(device)
    optimizer = opt.Adam(model.parameters(), lr=lr)
    #scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    
    #criterion = nn.MSELoss()   # 定义损失函数MSE
    #criterion = nn.L1Loss()
    criterion = nn.HuberLoss()
    criterion = criterion.to(device)

    x, y = [], []
    for epoch in range(num_epochs):
        progress_bar(epoch, num_epochs, prefix='process: ', length=50)
        model.train()  # 设置模型为训练模式
        for i,(inputs,true_y) in enumerate(train_iter):
            inputs = inputs.to(device)
            true_y = true_y.to(device) # 将输入和标签移至设备
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, true_y)  # 计算损失
            loss.backward()  # 反向传播:
            optimizer.step()  # 更新模型参数
        #scheduler.step()
        """
        model.eval()
        running_loss = 0.0
        for _,(inputs,true_y) in enumerate(test_iter):
            inputs = inputs.to(device)
            true_y = true_y.to(device) # 将输入和标签移至设备
            outputs = model(inputs)
            running_loss += criterion(outputs, true_y)
        running_loss /= test_iter.__len__()
        if epoch % 50 == 0:
            x.append(epoch)
            y.append(float(running_loss))
        """
    """
    plt.ylim(0, 0.5) 
    plt.plot(x, y, color="red")
    """
        
    running_loss = 0.0
    for j,(inputs,true_y) in enumerate(val_iter):
        inputs = inputs.to(device)
        true_y = true_y.to(device) # 将输入和标签移至设备
        outputs = model(inputs)
        running_loss += criterion(outputs, true_y)
    running_loss /= val_iter.__len__()
    print(f"\nThe end: running_loss: {running_loss}")
    checkpoint = {  
        'learning_rate': lr,
        'activate': activate,
        'size_list': size_list,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'model_state_dict': model.state_dict(),  
        'optimizer_state_dict': optimizer.state_dict(),   
        'loss': running_loss,  
    }
    filename = f"../check_points/checkpoint-0-N{N}.pth"
    counter = 0
    if os.path.exists(filename):
        while os.path.exists(f"../check_points/checkpoint-{counter}-N{N}.pth"):
            counter += 1 
        filename = f"../check_points/checkpoint-{counter}-N{N}.pth"
    torch.save(checkpoint, filename)
    plt.savefig(f"../loss/loss-{counter}-N{N}.png")
    print(f"Successfully save the checkpoint: {filename}")
    plt.clf()
    for _,(inputs,true_y) in enumerate(val_iter):
        inputs = inputs.to(device)
        true_y = true_y.to(device) # 将输入和标签移至设备
        outputs = model(inputs)
        plt.scatter([float(inputs[i]) for i in range(len(inputs))], [float(true_y[i]) for i in range(len(inputs))], color='blue')
        plt.scatter([float(inputs[i]) for i in range(len(inputs))], [float(outputs[i]) for i in range(len(inputs))], color='red')   
    filename = f'../pic/pic-eval-{counter}-N{N}.png'  
    plt.savefig(filename)
    print(f"Successfully save the eval figure: {filename}")

if __name__ == '__main__':  
    main()