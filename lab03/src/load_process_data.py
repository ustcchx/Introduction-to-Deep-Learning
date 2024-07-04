import  torch_geometric
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit

def load_dataset(device, name):
    dataset = torch_geometric.datasets.Planetoid(root=f"../tmp/{name}", name=name)
    return dataset[0].to(device.__str__()), dataset.num_node_features, dataset.num_classes

def process_data(mission_type: bool, data):
    if mission_type == 0:
        # 链路预测任务数据处理
        transform = RandomLinkSplit(is_undirected=True, num_test=0.2, num_val=0.2)
        train_data, val_data, test_data = transform(data)
        pos_edge_index = train_data.edge_label_index[:, 0: int(train_data.edge_label.size(0)/2)] #是单向的
        neg_edge_index = train_data.edge_label_index[:, int(train_data.edge_label.size(0)/2) + 1: ]
        val_pos_edge_index = val_data.edge_label_index[:, 0: int(val_data.edge_label.size(0)/2)] #是单向的
        val_neg_edge_index = val_data.edge_label_index[:, int(val_data.edge_label.size(0)/2) + 1: ]
        test_pos_edge_index = test_data.edge_label_index[:, 0: int(test_data.edge_label.size(0)/2)] #是单向的
        test_neg_edge_index = test_data.edge_label_index[:, int(test_data.edge_label.size(0)/2) + 1: ]
        return [pos_edge_index, neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index]
    if mission_type == 1:
        # 节点分类任务数据处理
        transform = RandomNodeSplit(num_test=0.2, num_val=0.2)
        trans_data = transform(data)
        return [trans_data.train_mask, trans_data.val_mask, trans_data.test_mask]
    
