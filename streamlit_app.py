import streamlit as st
import numpy as np
import cv2
import os
from predict import get_pred
from display import show_home_page, show_principle_page, show_about


class StreamlitShow:
    def __init__(self):
        self.images = []
        select_dict = {'首页': show_home_page, '模型原理': show_principle_page, '体验app': self.app_page, '关于': show_about}

        with st.sidebar:
            self.set_page = st.selectbox(
                '查看页面',
                select_dict.keys())

        st.title('🩺皮肤自动分类工具🧬')
        select_dict[self.set_page]()

    def app_page(self):
        # app页面的显示内容
        with st.sidebar:
            self.display_mode = st.radio(
                '选择一种查看图像的方式',
                ('原图', '只看病灶', '病灶范围', '模糊背景', '合成结果'))
            self.browse()

        # 当传入图像时，显示图像，各类别概率以及切换图像按钮
        if self.images:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                len_img = len(self.images)
                cur_cnt = self.get_cnt()
                last_button = st.button('上一张')
                if last_button:
                    with open('temporary files/select.txt', 'w') as f:
                        f.write(str((cur_cnt + len_img - 1) % len_img if len_img else 0))

            with col3:
                len_img = len(self.images)
                cur_cnt = self.get_cnt()
                next_button = st.button('下一张')
                if next_button:
                    with open('temporary files/select.txt', 'w') as f:
                        f.write(str((cur_cnt + len_img + 1) % len_img if len_img else 0))

            with col2:
                self.show()

        # 否则提示在左侧上传图像
        else:
            st.markdown('在左侧添加想要预测的图片即可查看预测结果哦')

    def get_cnt(self):
        # 当前所在图像被存储在temporary files中，由于每次按钮都会重新运行程序，所以页数保存在外部文件中
        if not os.path.exists('temporary files/select.txt'):
            with open('temporary files/select.txt', 'w') as f:
                f.write('0')
            return 0

        with open('temporary files/select.txt', 'r') as f:
            res = min(len(self.images) - 1, int(f.readline()))
        return res

    def show(self):
        if self.images:
            select_cnt = self.get_cnt()

            st.text('预测结果如下 当前第{}张 / 共{}张'.format(select_cnt+1, len(self.images)))
            image = self.images[select_cnt]

            # img_list存储了一系列的中间结果，可以用于显示中间信息，具体内容可以查看self.display_mode的相关代码
            res, img_list = get_pred(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if self.display_mode == '原图':
                st.image(image, channels="BGR")
            else:
                d = {'只看病灶': img_list[1], '病灶范围': img_list[2], '合成结果': img_list[0], '模糊背景': img_list[3]}
                show_image = cv2.resize(np.array(d[self.display_mode]), (image.shape[1], image.shape[0]))
                print(image.shape)
                st.image(show_image, channels="RGB")

            # 绘制概率条
            for i in range(len(res)):
                st.progress(res[i][0], text='{} (概率:{}%)'.format(res[i][1], '%.1f' % (res[i][0]*100)))

    # 上传文件使用，可同时上传多个文件，保存在self.images中
    def browse(self):
        available_type = ['jpg', 'png', 'jpeg', 'webp', 'bmp', 'tiff']
        uploaded_files = st.file_uploader("请上传需要预测的皮肤疾病图像", type=available_type, accept_multiple_files=True)

        st.write('你选择了如下图像')
        for uploaded_file in uploaded_files:
            # 将传入的文件转为Opencv格式
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            self.images.append(opencv_image)

            # 展示图片
            st.image(opencv_image, channels="BGR")


start = StreamlitShow()



