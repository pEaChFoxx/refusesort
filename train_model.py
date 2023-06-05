import tensorflow as tf
import matplotlib.pyplot as plt
from time import *


# 加载数据集，按照8比2的比例划分数据集，其中8份作为训练集，2份作为验证集
def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,  # 数据集划分比例
        subset="training",  # 选择训练集
        seed=123,
        color_mode="rgb",
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,  # 数据集划分比例
        subset="validation",  # 选择验证集
        seed=123,
        color_mode="rgb",
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    return train_ds, val_ds, class_names


# 模型加载，指定图片处理的大小和是否进行迁移学习
def model_load(IMG_SHAPE=(224, 224, 3), class_num=245):
    # 微调的过程中不需要进行归一化的处理
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')  # 加载mobilenetv2模型
    base_model.trainable = False  # 将主干的特征提取网络参数进行冻结
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
        # 归一化处理，将像素值处理为-1到1之间
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),  # 全局平均池化
        tf.keras.layers.Dense(class_num, activation='softmax')  # 设置最后的全连接层，用于分类
    ])
    model.summary()  # 输出模型信息
    # 模型训练
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])  # 用于编译模型，指定模型的优化器是adam优化器，模型的损失函数是交叉熵损失函数
    return model  # 返回模型


# 训练模型主流程
def train(epochs):
    begin_time = time()  # 记录开始时间
    train_ds, val_ds, class_names = data_load("C:/Users/RZY/Documents/trash_jpg/trash_jpg", 224, 224, 16)  # todo 记录数据集位置，修改为自己的数据集位置
    print(class_names)  # 输出类名
    model = model_load(class_num=len(class_names))  # 加载模型
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)  # 开始训练
    model.save("models/mobilenet_245_epoch30.h5")  # todo 修改了自己的模型保存位置，保存模型
    end_time = time()  # 记录开始时间
    run_time = end_time - begin_time  # 记录时间花费
    print('该循环程序运行时间：', run_time, "s")  # 输出程序运行时间


if __name__ == '__main__':
    train(epochs=30)  # todo 设置训练的轮数
