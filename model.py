# -*- coding: utf-8 -*-
"""
    model.py: CNN神经网络模型
"""
import tensorflow as tf

class CatsDogsModel(tf.keras.Model):
    def __init__(self, batch_size, n_classes):
        super(CatsDogsModel, self).__init__()
        # 保持原有的网络结构，但使用Keras层

        # 两个卷积层，每个卷积层后接一个池化层
        # 第一个卷积层conv1 使用 Conv2D 和 MaxPool2D 来提取图像特征。
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,  # 卷积核数量 提取16种不同的特征
            kernel_size=3,  # 卷积核大小为 3x3
            padding='same',  # 填充方式
            activation='relu'  # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),  # 池化窗口大小
            strides=2  # 步长为2，尺寸减半 由 208*208->104*104
        )

        # 第二个卷积层conv2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding='same',  # 保持尺寸不变 104*104
            activation='relu'
        )
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=1 # 因为步长为1，尺寸保持不变
        )

        # 两个全连接层
        self.flatten = tf.keras.layers.Flatten()                      # 使用 Flatten 将特征图展平。
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')   # 两个全连接层，使用 Dense 层进行分类。
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')

        # 输出层
        self.dense3 = tf.keras.layers.Dense(n_classes)  # n_classes=2  # 最后一个 Dense 层输出分类结果，类别数由 n_classes 参数决定。

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

