# 链路预测
import os
import torch
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt 
from torch_geometric.nn import Sequential
from torch.optim.lr_scheduler import StepLR
from torchmetrics import AUROC
from Conv_model import GCNConv
from load_process_data import load_dataset, process_data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取数据集
data, num_node_features, num_classes = load_dataset(device, 'Cora')
pos_edge_index, neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index = process_data(0, data)

class GCN(torch.nn.Module):

    def __init__(self, features, hidden_dimensions: list, classes):
        super(GCN, self).__init__()
        ### encode
        layers = []
        channel = features
        for s in hidden_dimensions:
            layers.append((GCNConv(channel, s), 'x, edge_index -> x'))
            channel = s
        layers.append((GCNConv(channel, classes), 'x, edge_index -> x'))
        self.convseq = Sequential('x, edge_index', layers)

        
    def forward(self, x, pos_edge_index, neg_edge_index):
        ### encode
        reversed_pos_edge_index = pos_edge_index[torch.tensor([1, 0]), :]
        pos_edge_index_mut = torch.cat([pos_edge_index, reversed_pos_edge_index], dim=-1)
        z = self.convseq(x, pos_edge_index_mut)
        ### decode
        selected_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[selected_edge_index[0]] * z[selected_edge_index[1]]).sum(dim=-1)
        return logits   

lr = 0.01
num_epochs = 400  # 设置训练轮数
decay_rate = 1
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
model = GCN(num_node_features, [512], num_classes)
model.to(device)
optimizer = opt.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=1, gamma=decay_rate)
x, y = [], []
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    optimizer.zero_grad()  # 清零梯度
    model_outputs = model(data.x, pos_edge_index, neg_edge_index)  # 前向传播
    labels =  torch.tensor(pos_edge_index.size(1)*[1.0] + neg_edge_index.size(1)*[0.0]).to(device)
    loss = criterion(model_outputs, labels)
    loss.backward()  # 反向传播:
    optimizer.step()  # 更新模型参数
    running_loss += loss.item()  # 累加损失\
    scheduler.step()
    model.eval()
    x.append(epoch)
    y.append(running_loss)
    if(epoch % 20 == 19):
        auc_metric = AUROC(task="binary")
        val_outputs = model(data.x, val_pos_edge_index, val_neg_edge_index)
        val_labels = torch.tensor(val_pos_edge_index.size(1)*[1.0] + val_neg_edge_index.size(1)*[0.0]).to(device)
        score = auc_metric(val_outputs, val_labels)
        print(f"epoch: {epoch}, running loss: {running_loss}, AUROC score: {score}")

counter = 0
filename = "../loss/loss-0.png"
if os.path.exists(filename):
    while os.path.exists(f"../loss/loss-{counter}.png"):
        counter += 1 
    filename = f"../loss/loss-{counter}.png"
plt.ylim(min(y), max(y)) 
plt.plot(x, y, color="red")
plt.savefig(f"../loss/loss-{counter}.png")

auc_metric = AUROC(task="binary")
test_outputs = model(data.x, test_pos_edge_index, test_neg_edge_index)
test_labels = torch.tensor(test_pos_edge_index.size(1)*[1.0] + test_neg_edge_index.size(1)*[0.0]).to(device)
score = auc_metric(test_outputs, test_labels)
print(f"Finall AUROC score(test data): {score}")
print(f"loss-{counter}.png is saved")
    




