## Lab01 代码使用说明
### 第一步：数据生成
```
python generate_data.py --N 10000
# 参数N: 代表数据集产生，默认为10000
```
### 第二步：模型训练与验证
#### 方式一：采用笔者已经调好的参数训练
```
python train.py --load_parameters "../check_points/parameters_N_200.pth" --N 200
# 以上导入N为200时的模型参数

python train.py --load_parameters "../check_points/parameters_N_2000.pth" --N 2000
# 以上导入N为2000时的模型参数

python train.py --load_parameters "../check_points/parameters_N_10000.pth" --N 10000
# 以上导入N为10000时的模型参数

# 加上--load_model 0代表仅导入超参数，即需要重新训练，默认为1，同时导入预训练模型参数
```

#### 方式二：超参数设置训练
```
python train.py
options:
  -h, --help            show this help message and exit
  --lr LR               输入学习率, default:3e-3
  --batch_size BATCH_SIZE
                        输入batch_size, default:2048
  --size_list SIZE_LIST [SIZE_LIST ...]
                        输入每层传递值的size， 
                        default:[1 ,8, 16 ,16 ,16 ,16 ,16 ,16 ,8, 1]， 
                        注意开头收尾必须为1
  --num_epochs NUM_EPOCHS
                        循环epoch，default:10000
  --activate ACTIVATE   激活函数类型: support ReLU/LeakyReLU/Tanh, default:ReLU
  --N N                 对应N个样本，default:10000
  --load_parameters LOAD_PARAMETERS
                        选择最佳参数的checkpoint的路径，如果设置为checkpoint文件的参数，其他参数传入会覆盖checkpoint的参数
  --load_model LOAD_MODEL
                        1：载入checkpoint model自身参数，0：不载入checkpoint model自身参数，需要重新训练
```
在模型训练完毕后，验证集也会自动完成验证，得到验证集的loss，其拟合的图像位于'/fig'文件目录下，模型的检查点也会相应记录在'/checkpoints'下。


### 第三步：模型测试
模型测试必须导入相应检查点文件的超参数，所以load_parameters是必需的。  
--load_parameters "../check_points/xxx.pth"
```
python test.py --load_parameters "../check_points/parameters_N_200.pth" --N 200
# 以上导入N为200时的模型参数

python test.py --load_parameters "../check_points/parameters_N_2000.pth" --N 2000
# 以上导入N为2000时的模型参数

python test.py --load_parameters "../check_points/parameters_N_10000.pth" --N 10000
# 以上导入N为10000时的模型参数

# 加上--load_model 0代表仅导入超参数，即需要重新训练，默认为导入预训练模型参数
```