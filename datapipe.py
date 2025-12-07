# datapipe.py 文件的注释版本
# 该文件负责数据管道，包括to_categorical函数、EmotionDataset类、数据归一化、获取数据、构建数据集和获取数据集函数。

import os  # 导入os模块
import numpy as np  # 导入numpy
import torch  # 导入PyTorch
from torch_geometric.data import Data, InMemoryDataset  # 导入图数据和内存数据集
from tqdm import tqdm  # 导入tqdm，用于进度条
import scipy.io as sio  # 导入scipy.io，用于加载mat文件
import glob  # 导入glob，用于文件匹配

subjects = 15  # 受试者数15
classes = 3  # 类别数3
version = 1  # 版本1

def to_categorical(y, num_classes=None, dtype='float32'):  # 定义转为one-hot函数
    y = np.array(y, dtype='int16')  # 转为int16数组，形状不变
    input_shape = y.shape  # 输入形状
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:  # 如果最后一维为1
        input_shape = tuple(input_shape[:-1])  # 去除最后一维
    y = y.ravel()  # 平铺为一维，形状 [total_elements]
    if not num_classes:  # 如果未指定类别数
        num_classes = np.max(y) + 1  # 最大值+1
    n = y.shape[0]  # 元素数
    categorical = np.zeros((n, num_classes), dtype=dtype)  # 初始化one-hot矩阵，形状 [n, num_classes]
    categorical[np.arange(n), y] = 1  # 设置对应位置为1
    output_shape = input_shape + (num_classes,)  # 输出形状
    categorical = np.reshape(categorical, output_shape)  # 重塑
    return categorical  # 返回one-hot


class EmotionDataset(InMemoryDataset):  # 定义情感数据集类，继承InMemoryDataset
    def __init__(self, stage, root, subjects, sub_i, X=None, Y=None, edge_index=None, transform=None, pre_transform=None):  # 初始化
        self.stage = stage  # 阶段（Train/Test）
        self.subjects = subjects  # 受试者数
        self.sub_i = sub_i  # 当前子索引
        self.X = X  # 输入数据
        self.Y = Y  # 标签
        self.edge_index = edge_index  # 边缘索引（未使用）
        super().__init__(root, transform, pre_transform)  # 调用父类
        self.data, self.slices = torch.load(self.processed_paths[0])  # 加载处理后的数据

    @property
    def processed_file_names(self):  # 处理文件名属性
        return ['./V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(version, self.stage, self.subjects, self.sub_i)]  # 返回格式化文件名

    def process(self):  # 处理函数
        data_list = []  # 数据列表
        num_samples = np.shape(self.Y)[0]  # 样本数
        for sample_id in tqdm(range(num_samples)):  # 遍历样本，带进度条
            x = self.X[sample_id, :, :]  # 提取x，形状 [62, features]
            x = torch.FloatTensor(x)  # 转为张量，形状不变
            y = torch.FloatTensor(self.Y[sample_id, :])  # 提取y，形状 [classes]
            data = Data(x=x, y=y)  # 创建Data对象（图数据，无边缘）
            data_list.append(data)  # 添加到列表
        data, slices = self.collate(data_list)  # 整理数据和切片
        torch.save((data, slices), self.processed_paths[0])  # 保存


def normalize(data):  # 定义归一化函数
    # data: 输入数据，形状 [samples, channels, features]
    mee = np.mean(data, 0)  # 均值，形状 [channels, features]
    data = data - mee  # 减均值，形状不变
    stdd = np.std(data, 0)  # 标准差，形状 [channels, features]
    data = data / (stdd + 1e-7)  # 除标准差，形状不变
    return data  # 返回归一化数据


def get_data():  # 获取数据函数
    path = 'E:/EEG_data/SEED/ExtractedFeatures/'  # 数据路径
    label = sio.loadmat(path + 'label.mat')['label']  # 加载标签，形状 [1, 15] 或类似
    sub_mov = []  # 受试者数据列表
    sub_label = []  # 标签列表
    files = sorted(glob.glob(os.path.join(path, '*')))  # 获取所有文件
    for sub_i in range(subjects):  # 遍历受试者
        sub_files = files[sub_i * 3: sub_i * 3 + 3]  # 每个受试者3个文件
        mov_data = []  # 电影数据列表
        for f in sub_files:  # 遍历文件
            data = sio.loadmat(f, verify_compressed_data_integrity=False)  # 加载mat
            keys = data.keys()  # 键列表
            de_mov = [k for k in keys if 'de_movingAve' in k]  # 提取de_movingAve键
            mov_datai = []  # 当前文件数据
            for t in range(15):  # 遍历15个trial
                temp_data = data[de_mov[t]].transpose(0, 2, 1)  # 转置，假设形状 [62, ?, 5] -> [62, 5, ?]
                data_length = temp_data.shape[-1]  # 长度
                mov_i = np.zeros((62, 5, 265))  # 初始化0数组，形状 [62, 5, 265]
                mov_i[:, :, :data_length] = temp_data  # 填充数据
                mov_i = mov_i.reshape(62, -1)  # 重塑 [62, 5*265=1325]
                mov_datai.append(mov_i)  # 添加
            mov_data.append(np.array(mov_datai))  # 添加到mov_data，形状 [15, 62, 1325]
        mov_data = np.vstack(mov_data)  # 垂直堆叠，形状 [45, 62, 1325] (3session*15)
        mov_data = normalize(mov_data)  # 归一化
        sub_mov.append(mov_data)  # 添加到sub_mov
        sub_label.append(np.hstack([label, label, label]).squeeze())  # 重复标签3次，形状 [45]
    sub_mov = np.array(sub_mov)  # 转为数组，形状 [15, 45, 62, 1325] 15人,3session*15trial,62channel,5DE*265s
    sub_label = np.array(sub_label)  # [15, 45]
    return sub_mov, sub_label  # 返回


def build_dataset(subjects):  # 构建数据集函数
    load_flag = True  # 加载标志
    for sub_i in range(subjects):  # 遍历受试者
        path = './processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(version, 'Train', subjects, sub_i)  # 路径
        if not os.path.exists(path):  # 如果不存在
            if load_flag:  # 如果需加载
                mov_coefs, labels = get_data()  # 获取数据，mov_coefs [15, 45, 62, 1325], labels [15, 45]
                used_coefs = mov_coefs  # 使用
                load_flag = False  # 设置已加载
            index_list = list(range(subjects))  # 索引列表
            del index_list[sub_i]  # 删除当前
            test_index = sub_i  # 测试索引
            train_index = index_list  # 训练索引
            X = used_coefs[train_index, :].reshape(-1, 62, 1325)  # 训练X，形状 [14*45=630, 62, 1325]
            Y = labels[train_index, :].reshape(-1)  # 训练Y，形状 [630]
            testX = used_coefs[test_index, :].reshape(-1, 62, 1325)  # 测试X，形状 [45, 62, 1325]
            testY = labels[test_index, :].reshape(-1)  # 测试Y，形状 [45]
            _, Y = np.unique(Y, return_inverse=True)  # 标签编码，Y [630]
            Y = to_categorical(Y, classes)  # one-hot，形状 [630, 3]
            _, testY = np.unique(testY, return_inverse=True)  # 同上
            testY = to_categorical(testY, classes)  # [45, 3]
            train_dataset = EmotionDataset('Train', './', subjects, sub_i, X, Y)  # 创建训练数据集
            test_dataset = EmotionDataset('Test', './', subjects, sub_i, testX, testY)  # 测试数据集


def get_dataset(subjects, sub_i):  # 获取数据集函数
    train_path = f'./processed/V_{version}_Train_CV{subjects}_{sub_i}.dataset'  # 训练路径
    target_sub = sub_i  # 目标子
    target_path = f'./processed/V_{version}_Test_CV{subjects}_{target_sub}.dataset'  # 目标路径
    print("train_path ", train_path)  # 打印
    print("target_path ", target_path)
    train_dataset = EmotionDataset('Train', './', subjects, sub_i)  # 创建训练数据集（加载已处理）
    target_dataset = EmotionDataset('Test', './', subjects, target_sub)  # 目标数据集

    return train_dataset, target_dataset  # 返回