# streamlit-skin-classification
### 前言
这是一个简单的皮肤分类网页端app，基于streamlit实现。
其主要方式是通过先分割再分类完成的。
你可以在[这里](https://g6z0955446.yicp.fun/)体验这个网页app

### 使用这个项目
你可以在这里下载已经训练好的模型
[分割模型权重](https://huggingface.co/Inubashiri/IENet/resolve/main/v8_isic.pth)
[分类模型权重](https://huggingface.co/Inubashiri/IENet/resolve/main/my_model(199).pkl)
之后打开cmd窗口，运行
> streamlit run streamlit_app.py

### 或者
你可以使用自己的分类模型，对自己想要分类的内容进行分类
这里你需要在prediction.py中修改
输入的应当是一张numpy格式的图像
输出是一个二维列表，包含这个预测的多个类别以及对应的置信度
例如:
[[0.6, 类别A], [0.2, 类别B], [0.1, 类别C]]

### 联系作者
你可以通过如下方式和我取得联系:
邮箱: 705172578@qq.com
微信: dbx19980106
