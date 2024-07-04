## Lab02代码使用说明
### 0. python环境说明
~~~
torch GPU版本
tqdm 
gradio == 3.41.2
# 这里限制gradio版本是因为代码中加入了与该版本前端容器匹配的外部CSS样式
~~~
### 1. 命令行进入src目录后，启动程序gradio UI
~~~
python run_web.py
~~~
得到如下类似URL：
~~~
Running on local URL:  http://127.0.0.1:7860
~~~
在浏览器中打开即可

### 2. 选择训练 or 测试模式
如果选择训练模式，则有两种训练方法，一种是重新从头开始训练，此时不需要导入checkpoint文件（为空即可），但是需要调整batch size, learning rate, max epoch等超参数；另一种是导入checkpoint文件，从该记录点开始训练。笔者在训练集上训练出的最佳模型参数全部记录在checkpoint-block-2.pth（18层模型，每个残差块含有的block个数需要为2）与checkpoint-block-3.pth（26层模型，每个残差块含有的block个数需要为3）文件中，可以从这两个记录点开始再次训练。

测试模式只需要将之前训练得到的pth文件导入即可得到在测试集上的准确率。


### 3. 得到结果
训练测试结束后，check_points文件夹中存有当此训练表现最佳的模型内部参数，loss中存有训练损失变化图像，同时也会给出训练后的模型在测试集上的准确率。