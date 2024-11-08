# -*- coding: utf-8 -*-
"""
    training.py: 模型的训练及评估
"""
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import input_data
from model import CatsDogsModel

# 保持原有的参数设置
N_CLASSES = 2      # 二分类任务（猫和狗）
IMG_W = 208        # 图片宽度
IMG_H = 208        # 图片高度
BATCH_SIZE = 16    # 每批处理16张图片
CAPACITY = 2000    # 队列容量
MAX_STEP = 1000    # 训练1000步
learning_rate = 0.0001  # 学习率

# 使用相对路径
train_dir = os.path.join('data', 'train')
logs_train_dir = 'log'

# 获取训练数据和标签
train_img, train_label = input_data.get_files(train_dir)

# 创建训练数据集
train_dataset = input_data.create_dataset(
    train_img, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY
)

# 创建模型实例
model = CatsDogsModel(BATCH_SIZE, N_CLASSES)

# 定义优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 定义损失函数
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 定义准确率度量
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()


# 训练步骤
@tf.function  # 使用tf.function装饰器加速训练
def train_step(images, labels):
    with tf.GradientTape() as tape:  # 记录梯度
        predictions = model(images, training=True)  # 前向传播
        loss = loss_fn(labels, predictions)        # 计算损失

    # 计算梯度并更新模型参数
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 更新准确率指标
    train_acc_metric.update_state(labels, predictions)
    return loss


# 记录训练数据用于绘图
accuracy_list = []   # 存储每100步的准确率
loss_list = []      # 存储每100步的损失值
step_list = []      # 存储训练步数

# 训练循环
for step in range(MAX_STEP):
    for images, labels in train_dataset:
        loss = train_step(images, labels)

        # 每50步打印一次训练信息
        if step % 50 == 0:
            print(f'Step {step}, train loss = {loss:.2f}, '
                  f'train accuracy = {train_acc_metric.result().numpy() * 100:.2f}%')

        # 每100步记录一次数据用于绘图
        if step % 100 == 0:
            accuracy_list.append(train_acc_metric.result().numpy())  # 记录当前准确率
            loss_list.append(loss.numpy())                           # 记录当前损失值
            step_list.append(step)                                   # 记录当前步数

        # 定期保存模型
        if step % 5000 == 0 or (step + 1) == MAX_STEP:
            model.save_weights(os.path.join(logs_train_dir, f'model_step_{step}'))

# 绘制训练过程图
plt.figure()  # 创建新的图形窗口
# 绘制准确率曲线（蓝色实线）
plt.plot(step_list,           # x轴：训练步数
         accuracy_list,       # y轴：准确率值
         color='b',          # 蓝色
         label='cnn_accuracy' # 图例标签
)

# 绘制损失值曲线（红色虚线）
plt.plot(step_list,          # x轴：训练步数
         loss_list,          # y轴：损失值
         color='r',          # 红色
         label='cnn_loss',   # 图例标签
         linestyle='dashed'  # 虚线样式
)

plt.xlabel("Step")           # x轴标签：训练步数
plt.ylabel("Accuracy/Loss")  # y轴标签：准确率/损失值
plt.legend()                 # 显示图例
plt.show()                   # 显示图形
