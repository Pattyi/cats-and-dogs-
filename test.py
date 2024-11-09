# -*- coding: utf-8 -*-
"""
    test.py: 用训练好的模型对随机一张图片进行猫狗预测
"""
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def get_one_image(img_list):
    """
        输入：
            img_list：图片路径列表 存放的是每个图片完整的路径 data/test/6.jpg
        返回：
            image：从图片路径列表中随机挑选的一张图片，格式为NumPy格式
    """
    n = len(img_list)  # 获取文件夹下图片的总数
    ind = np.random.randint(0, n)  # 从 0~n 中随机选取下标
    img_dir = img_list[ind]  # 根据下标得到一张随机图片的路径 data/test/6.jpg

    image = Image.open(img_dir)  # 打开img_dir路径下的图片 PIL格式
    image = image.resize([208, 208])  # 改变图片的大小，定为宽高都为208像素
    image = np.array(image)  # 转成多维数组，向量的格式 由PIL格式->NumPy格式
    return image


def process_image(image):
    """
    处理图片函数：标准化处理
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, 0)  # 增加batch维度
    return image


def evaluate_one_image():
    """
    测试函数
    """
    # 修改成自己测试集的文件夹路径
    test_dir = os.path.join('data', 'test')

    # 获取测试图片的路径列表
    from input_data import get_files
    test_img = get_files(test_dir)[0]  # get_files()[0]得到的是图片路径的列表

    # 从测试集中随机选取一张图片
    image_array = get_one_image(test_img)  # 传入图片路径列表，得到numpy类型的一张图片

    # 创建模型
    # 修改后的代码（推荐）：
    from model import CatsDogsModel
    model = CatsDogsModel(batch_size=1, n_classes=2)  # batch_size=1因为只预测一张图片调用__init__(self, batch_size, n_classes)

    # 加载训练好的模型权重
    checkpoint_path = os.path.join('log', 'model.ckpt-9999')  # 修改为您的模型路径
    if os.path.exists(checkpoint_path + '.index'):
        print("从指定路径中加载模型...")
        model.load_weights(checkpoint_path)
        print('模型加载成功')
    else:
        print('模型加载失败，checkpoint文件没找到！')
        return

    # 处理单张图片
    processed_image = process_image(image_array)

    # 进行预测
    prediction = model(processed_image, training=False)  # 调用call(self, inputs, training=None)
    prediction = tf.nn.softmax(prediction)

    # 获取预测结果
    max_index = tf.argmax(prediction, axis=1)
    score = tf.reduce_max(prediction, axis=1)

    if max_index[0] == 0:
        print('图片是猫的概率为： %.2f%%' % (score[0] * 100))
    else:
        print('图片是狗的概率为： %.2f%%' % (score[0] * 100))

    # 显示测试图片
    plt.imshow(image_array)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # 调用方法，开始测试
    evaluate_one_image()
