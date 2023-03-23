import streamlit as st
import cv2


def show_home_page():
    st.markdown('#### 👋欢迎！')
    st.markdown('这是一款简易的皮肤分类工具，他可以对你输入皮肤疾病图像进行自动分类，并给出排名前三的分类概率。')
    st.markdown('目前模型收录的皮肤疾病种类如下：')
    names = ['光化性角化病', '基底细胞癌', '良性角化病', '皮肤纤维瘤', '黑色素瘤', '黑色素细胞性痣', '血管性皮肤病变']
    english_names = ['Actinic keratoses', 'Basal cell carcinoma',  'Benign keratosis',  'Dermatofibroma',  'Melanoma', 'Vascular skin lesions']
    for name, english_name in zip(names, english_names):
        st.markdown(f'- {name}({english_name})')
    st.markdown('你可以通过左边导航栏，查看这个工具的原理，或者体验一下网页端的app😊!')
    st.text('')
    st.write('🤷‍♂️:red[如果出现报错信息  “Failed to execute ‘removeChild‘ on ‘Node‘: The node to be removed is not a child of this node” ,请刷新页面重试或者换个浏览器。]')


def show_principle_page():
    st.markdown('#### 1. 处理流程')
    st.markdown('工具的处理流程如下图所示，在获取到图像后会先使用IENet进行分割，通过分割结果，获取病灶区域，轮廓，并对背景进行均值滤波, 之后将这三种结果融合并传到分类网络中进行分类。')
    src = cv2.imread('E:\streamlit_skin\display\process.png')
    st.image(src, channels='BGR')
    st.text('')

    st.markdown('#### 2. 分割模型')
    st.markdown('分割模型的结构如下图所示')
    src = cv2.imread('E:\streamlit_skin\display\ienet.png')
    st.image(src, channels='BGR')
    st.markdown('提出的网络可分为四部分:')
    st.markdown('**（1）输入增强部分**: 可从不同尺寸的图像中获取特征，以弥补在特征提取网络中图像丢失的信息。')
    st.markdown('**（2）编码器部分**: 以带预训练权重的ResNeXt50为骨干网络，结合输入增强部分的提取图像的特征。')
    st.markdown('**（3）融合注意部分**: 通过融合相邻层次之间的信息，将网络上下层的信息整合到一起进行多尺度的特征提取并分配注意力，最终将特征传给解码器部分。')
    st.markdown('**（4）解码器部分**: 通过UpSample操作和Inception Block将编码器部分产生的最高语义信息进行上采样，结合融合注意部分的输出获的显著性图。其整体结构图如图1所示。')
    st.text('')

    st.markdown('#### 3. 分类模型')
    st.markdown('分类模型使用DaViT实现，优化器使用Lion，具体可参考下方链接:')
    st.markdown('**分类器:DaViT**')
    st.markdown('- 论文: [DaViT: Dual Attention Vision Transformer (ECCV 2022)](https://arxiv.org/pdf/2204.03645.pdf)')
    st.markdown('- 项目: [https://github.com/dingmyu/davit](https://github.com/dingmyu/davit)')

    st.markdown('**优化器:Lion**')
    st.markdown('- 论文: [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675v2)')
    st.markdown('- 项目: [https://github.com/google/automl/tree/master/lion](https://github.com/google/automl/tree/master/lion)')


def show_about():
    st.markdown('#### 关于项目🗂️')
    st.markdown('该项目代码目前发布在github上，链接如下:')
    st.markdown('[https://github.com/d705172578/streamlit-skin-classification](https://github.com/d705172578/streamlit-skin-classification)')

    st.markdown('#### 关于作者👨‍🎓')
    st.markdown('想了解更多该项目信息，可以联系作者')
    st.markdown('邮箱📧: 705172578@qq.com')
    st.markdown('微信💬: dbx19980106')
