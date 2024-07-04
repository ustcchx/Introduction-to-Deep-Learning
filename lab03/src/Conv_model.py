import torch
import torch.nn.functional as nnFun
import torch.nn as nn
import dgl

def PairNorm(x_feature):
    mode = 'PN-SI'
    scale = 1
    col_mean = x_feature.mean(dim=0)
    if mode == 'PN':
        x_feature = x_feature - col_mean
        row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
        x_feature = scale * x_feature / row_norm_mean

    if mode == 'PN-SI':
        x_feature = x_feature - col_mean
        row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
        x_feature = scale * x_feature / row_norm_individual

    if mode == 'PN-SCS':
        row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
        x_feature = scale * x_feature / row_norm_individual - col_mean

    return x_feature

class GCNConv(torch.nn.Module):
    
    def __init__(self, in_channel, out_channel, device = 'cuda'):
        super(GCNConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight = nn.Parameter(torch.randn(in_channel, out_channel))    
        self.device = device
    
    def __repr__(self):
        return f'GCNConv({self.in_channel}, {self.out_channel})'
    
    def forward(self, x, edge_index):
        
        g = dgl.DGLGraph().to('cuda')
        g.add_nodes(x.size(0))
        #g = dgl.graph((edge_index[0], edge_index[1])).to(self.device)
        g.add_edges(edge_index[0], edge_index[1])
        g = g.add_self_loop()
        # add自环
        transform = dgl.DropEdge(p = 0)
        # 随机丢弃
        g = transform(g)
        g.ndata['h'] = torch.mm(x, self.weight)
        degs = g.out_degrees().float()  # D
        norm = torch.pow(degs, -0.5)  # D^{-1/2}
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)  
        g.update_all(self.gcn_msg, self.gcn_reduce)
        x = nnFun.normalize(g.ndata['h'], p=2, dim=1)
        # x = PairNorm(x) # 添加pairnorm
        # x = nnFun.relu(x)
        x = nnFun.tanh(x)
        # leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        # x = leaky_relu(x)
        return x
    
    def gcn_msg(self, edge):
        msg = edge.src['h'] * edge.src['norm']
        # msg = edge.src['h']
        return {'m': msg}

    # 定义消息聚合规则
    def gcn_reduce(self, node):
        accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
        return {'h': accum}