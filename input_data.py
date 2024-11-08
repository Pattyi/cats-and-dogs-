# -*- coding: utf-8 -*-
"""
    input_data.py: 读取训练数据
"""
import tensorflow as tf
import numpy as np
import os


def get_files(file_dir):
    """
        输入：
            file_dir：存放训练图片的文件地址
        返回:
            image_list：乱序后的图片路径列表
            label_list：乱序后的标签(相对应图片)列表
    """
    # 原有代码保持不变，直到return语句
    # 建立空列表
    cats = []  # 存放是猫的图片路径地址
    label_cats = []  # 对应猫图片的标签
    dogs = []  # 存放是猫的图片路径地址
    label_dogs = []  # 对应狗图片的标签

    # 从file_dir路径下读取数据，存入空列表中
    # The Python os.listdir() method returns a list containing the names of the files within the given directory.
    for file in os.listdir(file_dir):  # file就是要读取的图片带后缀的文件名
        # os.listdir(file_dir) 返回的结果，返回一个列表，列表内容为所给文件路径data/train里的文件的名字：
        # ['cat.0.jpg',
        # 'cat.1.jpg',
        # 'cat.10.jpg',
        # 'cat.2.jpg'...]
        name = file.split(sep='.')  # 图片格式是cat.1.jpg, 处理后name为['cat', '0', 'jpg']
        if name[0] == 'cat':  # name[0]获取图片名
            cats.append(os.path.join(file_dir,file))  # 若是cat，则将该图片路径地址添加到cats数组里['data\\train\\cat.0.jpg', 'data\\train\\cat.1.jpg',...]
            label_cats.append(0)  # 并且对应的label_cats添加0标签 （这里记作：0为猫，1为狗）[0, 0, 0, ..., 0, 0]
        else:
            dogs.append(os.path.join(file_dir, file))
            label_dogs.append(1)  # 注意：这里添加进的标签是字符串格式，后面会转成int类型

    # print('There are %d cats\nThere are %d dogs' % (len(cats), len(dogs)))

    # 这里把猫狗图片及标签合并分别存在image_list和label_list
    # np.hstack() 会将输入的列表转换为NumPy数组并在水平方向合并
    image_list = np.hstack((cats, dogs))  # 在水平方向平铺合成一个行向量，即两类地址的拼接->合并成一个存放地址的NumPy数组和一个存放标签的NumPy数组
    label_list = np.hstack((label_cats, label_dogs))  # [0, 0, ...,0 ,1, 1, ...,1]

    temp = np.array([image_list, label_list])  # 生成一个2*25000的数组，即2行、25000列
    temp = temp.transpose()  # 转置向量，大小变成25000 * 2
    np.random.shuffle(temp)  # 乱序，打乱这25000行排列的顺序 第一列是地址 第二列是标签 一二列数据相互对应
    # 为什么要合并且乱序？

    # 通过 list() 函数将NumPy数组转回列表 由一个NumPy数组又变成两个列表
    # 列表(list) -> NumPy数组(numpy.ndarray) -> 列表(list)
    image_list = list(temp[:, 0])  # 所有行，列=0（选中所有猫狗图片路径地址），即重新存入乱序后的猫狗图片路径
    label_list = list(temp[:, 1])  # 所有行，列=1（选中所有猫狗图片对应的标签），即重新存入乱序后的对应标签
    label_list = [int(float(i)) for i in label_list]  # 把标签列表转化为int类型（用列表解析式迭代，相当于精简的for循环）

    return image_list, label_list


def process_image(image_path, image_W, image_H):
    """处理单个图片"""
    # 读取图片
    img = tf.io.read_file(image_path)
    # 解码图片（使用新版API）
    img = tf.io.decode_jpeg(img, channels=3)
    # 调整大小
    img = tf.image.resize(img, [image_W, image_H])
    # 标准化
    # img = tf.cast(img, tf.float32) 多余 不需要额外的类型转换步骤 per_image_standardization会输出float32
    img = tf.image.per_image_standardization(img)
    return img


def create_dataset(image_list, label_list, image_W, image_H, batch_size, capacity):
    """
        创建数据集
        替代原有的get_batch函数
    """
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))

    # 预处理图片 对数据集进行映射（map）操作
    dataset = dataset.map( # 图片路径(x) -> 读取文件 -> 解码为图像 -> 调整大小 -> 标准化 -> 返回处理后的图像张量
        lambda x, y: (process_image(x, image_W, image_H), y), # 映射函数 转换函数
        # 这里的返回值是一个元组，包含两个元素：
        # process_image(x, image_W, image_H) - 处理后的图片
        # y - 对应的标签
        num_parallel_calls=tf.data.AUTOTUNE # 并行处理设置
    )

    # 打乱数据
    dataset = dataset.shuffle(buffer_size=capacity)

    # 分批
    dataset = dataset.batch(batch_size)

    # 预取数据
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
