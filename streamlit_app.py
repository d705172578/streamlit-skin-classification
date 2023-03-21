import streamlit as st
import numpy as np
import cv2
import os
from predict import get_pred


class StreamlitShow:
    def __init__(self):
        self.images = []

        st.title('皮肤疾病自动分类工具')
        with st.sidebar:
            self.browse()

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
        else:
            st.text('在左侧添加想要预测的图片即可查看预测结果哦')

    def get_cnt(self):
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
            st.image(image, channels="BGR")
            res = get_pred(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            for i in range(3):
                st.progress(res[i][0], text='{} (概率:{}%)'.format(res[i][1], '%.1f' % (res[i][0]*100)))

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



