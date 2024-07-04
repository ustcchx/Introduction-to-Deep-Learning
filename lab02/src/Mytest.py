from get_data import CIFAR10_Dataset
import CNN
from torch.utils.data import DataLoader
import torch

def test_func(path: str, block_layer_num: int):
    test_data = CIFAR10_Dataset("test")
    checkpoint = torch.load(f"..\check_points\{path}")
    resnet18_model=CNN.ResNet([block_layer_num]*4, 10, 0)
    resnet18_model.load_state_dict(checkpoint['model_state_dict']) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 256
    test_loader = DataLoader(test_data, batch_size=batch_size)
    model = resnet18_model
    model.to(device)
    model.eval()
    acc = 0
    for j,(inputs,labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移至设备
        outputs = model(inputs)
        for k in range(len(outputs)) :
            s = torch.argmax(outputs[k])
            if s == labels[k] :
                acc += 1
    acc_rate = acc / test_data.__len__()
    return [acc_rate]

    