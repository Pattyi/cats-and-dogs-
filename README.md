## 使用CNN（卷积神经网络）进行猫狗图片分类的项目
**主要功能：**
* 数据处理：将猫狗图片处理成统一大小（208x208）
* 模型训练：使用CNN网络训练分类器
* 结果预测：能够预测一张图片是猫还是狗的概率
  
**特点：**
* 使用了TensorFlow2框架
* 支持GPU加速
* 包含完整的训练和测试流程
* 有可视化功能，方便观察训练过程

**这是一个完整的深度学习项目示例，展示了从数据处理到模型训练再到预测的完整流程。**
## 主要包含四个主要文件：
* input_data.py: 数据预处理模块
* model.py: CNN模型定义
* training.py: 模型训练
* test.py: 模型测试
## 四个主要文件介绍
**input_data.py: 数据预处理模块**
- 读取训练数据和标签,将图片数据转换为适合神经网络处理的格式def get_files(file_dir)
- 数据预处理,处理单个图片（调整大小、标准化）def process_image(image_path, image_W, image_H)
- 创建数据集，包括数据增强和批处理def create_dataset(image_list, label_list, image_W, image_H, batch_size, capacity)

**model.py: CNN模型定义**

- CatsDogsModel类：
  - 两个卷积层（每层16个卷积核）
  - 两个池化层（降维）
  - 两个全连接层（特征提取）
  - 一个输出层（分类）

**training.py: 模型训练**
- 设置训练参数（批次大小、学习率等）
- 创建和训练模型
- 记录训练过程（损失值、准确率）
- 保存训练模型
- 可视化训练结果

**test.py: 模型测试**
* 从测试集随机选择一张图片
* 加载训练好的模型
* 对选中的图片进行预测
* 显示预测结果和图片
## 初学者学习步骤：
* 先理解整体项目结构
* 逐个文件深入学习代码实现
* 尝试修改参数观察效果
* 尝试添加新功能或改进
## 项目结构
<img width="320" alt="image" src="https://github.com/user-attachments/assets/99050cdd-6d6d-40d0-a610-180a589f1441">

## 文件说明
* data(文件夹)：包含 test 测试集和 train 训练集

  我用夸克网盘分享了「data」文件夹。
链接：https://pan.quark.cn/s/daa6c9c7f9c8
* log(文件夹)：保存训练模型和参数
* input_data.py：数据预处理模块，为其他模块提供数据
* model.py：负责实现我们的神经网络模型
* training.py：负责实现模型的训练以及评估 【1.先跑这个来训练好模型，再跑test.py】
* test.py： 从测试集中随机抽取一张图片, 进行预测是猫还是狗  【2.跑完training.py后，再跑这个来测试图片进行预测猫或狗】

