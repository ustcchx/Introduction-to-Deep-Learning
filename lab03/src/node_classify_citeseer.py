# 节点分类
import os
import torch
import matplotlib.pyplot as plt 
import torch.nn.functional as nnFun
import torch.nn as nn
import torch.optim as opt
from torch_geometric.nn import Sequential
from torch.optim.lr_scheduler import StepLR
from Conv_model import GCNConv
from load_process_data import load_dataset, process_data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取与处理数据集
data, num_node_features, num_classes = load_dataset(device, 'citeseer')
train_mask, val_mask, test_mask = process_data(1, data)

#构造结点分类的GCN
class GCN(torch.nn.Module):

    def __init__(self, features, hidden_dimensions: list, classes):
        super(GCN, self).__init__()
        
        layers = []
        channel = features
        for s in hidden_dimensions:
            layers.append((GCNConv(channel, s), 'x, edge_index -> x'))
            channel = s
        layers.append((GCNConv(channel, classes), 'x, edge_index -> x'))
        self.convseq = Sequential('x, edge_index', layers)
        
    def forward(self, data):
        x = self.convseq(data.x, data.edge_index)
        return nnFun.log_softmax(x, dim=1)
   
def accuracy(y_pred, y_true):
    """Calculate accuracy."""
    return torch.sum(y_pred == y_true) / len(y_true)
    

lr = 0.005
num_epochs = 400  # 设置训练轮数
decay_rate = 1
criterion = nn.CrossEntropyLoss()
# model = GCN(num_node_features, [512, 512, 512], num_classes)
model = GCN(num_node_features, [512], num_classes)
model.to(device)
optimizer = opt.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=1, gamma=decay_rate)
x, y = [], []
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    optimizer.zero_grad()  # 清零梯度
    model_outputs = model(data)  # 前向传播
    loss = criterion(model_outputs[train_mask], data.y[train_mask])
    loss.backward()  # 反向传播:
    optimizer.step()  # 更新模型参数
    running_loss += loss.item()  # 累加损失\
    scheduler.step()
    model.eval()
    x.append(epoch)
    y.append(running_loss)
    if(epoch % 20 == 19):
        val_acc = accuracy(model_outputs[val_mask].argmax(dim=1), data.y[val_mask])
        print(f"epoch: {epoch}, running loss(train data): {running_loss}, Accuracy(val data): {val_acc}")

counter = 0
filename = "../loss/loss-0.png"
if os.path.exists(filename):
    while os.path.exists(f"../loss/loss-{counter}.png"):
        counter += 1 
    filename = f"../loss/loss-{counter}.png"
plt.ylim(min(y), max(y)) 
plt.plot(x, y, color="red")
plt.savefig(f"../loss/loss-{counter}.png")

test_acc = accuracy(model_outputs[test_mask].argmax(dim=1), data.y[test_mask])
print(f"Finnal Accuracy(test data): {test_acc}")
print(f"loss-{counter}.png is saved")


