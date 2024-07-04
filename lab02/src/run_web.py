import gradio as gr
from train import train_func

with open('style.css', 'r', encoding="UTF-8") as file:  
    custom_css = file.read()

inter1 = gr.Interface(
        fn=train_func,
        inputs=[
            gr.Textbox(value='checkpoint-block-3.pth', label="导入checkpoint路径，注意网络结构是否一致，可填"),
            gr.Textbox(value="0.0005", label="学习率"),
            gr.Slider(value=256, minimum=0, maximum=512, step=1, label="batch size*"),
            gr.Slider(value=100, minimum=0, maximum=500, step=1, label="最大epoch*"),
            gr.Slider(value=0, minimum=0, maximum=1, step=0.01, label="dropout对应的随机概率*"),
            gr.Slider(value=0.95, minimum=0, maximum=1, step=0.01, label="StepLR对应的decay rate*"),
            gr.Radio([2, 3], label="每个残差块含有的block个数*", value = 3),
            gr.Radio(["train", "test"], label="选择模式*，选择test模式后超参数设置就没有效果，将只根据pth文件算出测试集上的准确率", value = "train"),
        ], 
        outputs=[ 
            gr.Textbox(label="在验证数据集上的最优准确率"),
            gr.Textbox(label="在验证集上表现最优准确率的模型对应的epoch"),
            gr.Textbox(label="本次训练中在验证集上表现最优异的模型参数存入以下路径"),
            gr.Textbox(label="在测试集上取得的准确率"),
            gr.Image(label="每一个epoch损失展示"),
        ],
        css=custom_css
) 
inter1.launch()