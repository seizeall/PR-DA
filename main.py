# main.py 文件的注释版本
# 该文件是主程序，负责设置随机种子、定义原型对比损失函数、训练和评估函数，以及主函数中的交叉验证循环。
# 包括数据集构建、模型训练、评估、结果可视化和保存。

import os  # 导入os模块，用于文件和目录操作
import numpy as np  # 导入numpy，用于数值计算
import pandas as pd  # 导入pandas，用于数据帧处理
import torch  # 导入PyTorch核心库
import torch.nn.functional as F  # 导入PyTorch函数模块，用于激活函数等
from torch_geometric.data import DataLoader  # 从torch_geometric导入DataLoader，用于图数据加载
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve, auc  # 从sklearn导入评估指标
from datapipe import build_dataset, get_dataset  # 从datapipe导入数据集构建函数
from Net import PR_DA  # 从Net导入PR_DA模型
import random  # 导入random，用于随机数生成
import matplotlib.pyplot as plt  # 导入matplotlib，用于绘图
import seaborn as sns  # 导入seaborn，用于热图等可视化
from matplotlib.colors import LinearSegmentedColormap  # 导入颜色映射，用于自定义颜色


def set_random_seed(seed=42):  # 定义设置随机种子的函数，确保实验可重复
    np.random.seed(seed)  # 设置numpy随机种子
    random.seed(seed)  # 设置Python内置随机种子
    torch.manual_seed(seed)  # 设置PyTorch随机种子
    torch.cuda.manual_seed(seed)  # 设置CUDA随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有CUDA设备的随机种子
    torch.backends.cudnn.deterministic = True  # 使cuDNN确定性行为
    torch.backends.cudnn.benchmark = False  # 禁用cuDNN基准测试以确保确定性


subjects = 15  # 定义受试者数量为15
epochs = 200  # 定义训练轮数为200，可以调整
classes = 3  # 定义类别数为3（情感类别）
Network = PR_DA  # 指定网络模型为PR_DA
device = torch.device('cuda', 0)  # 指定设备为CUDA的第0个GPU
version = 1  # 版本号初始为1
lambda_domain_weight = 0.1  # 域损失权重为0.1
lambda_contrastive_weight = 0.1  # 对比损失权重为0.1
temperature_cl = 0.1  # 对比学习温度参数为0.1
lambda_proto_sep_weight = 0.05  # 原型分离损失权重为0.05
initial_bilinear_weight = 0.0  # 双线性权重初始值为0.0
max_bilinear_weight = 0.5  # 双线性权重最大值为0.5
bilinear_warmup_epochs = epochs / 2  # 双线性权重预热轮数为总轮数的一半，即100

set_random_seed(42)  # 调用函数设置随机种子为42

# 创建结果文件夹
os.makedirs('./result', exist_ok=True)  # 如果result文件夹不存在，则创建

while True:  # 循环查找未使用的日志文件版本
    dfile = f'./result/LOG_{version:.0f}.csv'  # 格式化日志文件名
    if not os.path.exists(dfile):  # 如果文件不存在，跳出循环
        break
    version += 1  # 否则版本号加1

df = pd.DataFrame()  # 创建空DataFrame
df.to_csv(dfile)  # 保存为空CSV文件


def prototype_contrastive_loss_fn(features, labels, prototypes, temperature, lambda_ps_weight, device):  # 定义原型对比损失函数
    # features: 特征向量，形状 [batch_size, feature_dim]
    # labels: 标签，形状 [batch_size]
    # prototypes: 原型向量，形状 [num_classes, feature_dim]
    # temperature: 温度参数
    # lambda_ps_weight: 原型分离权重
    # device: 设备

    valid_prototype_mask = torch.norm(prototypes, dim=1) > 1e-8  # 计算有效原型掩码，形状 [num_classes]
    if valid_prototype_mask.sum() == 0:  # 如果没有有效原型，返回0损失
        return torch.tensor(0.0, device=device)

    valid_prototypes = prototypes[valid_prototype_mask]  # 提取有效原型，形状 [num_valid, feature_dim]
    original_indices_of_valid_prototypes = torch.arange(prototypes.shape[0], device=device)[valid_prototype_mask]  # 有效原型的原始索引，形状 [num_valid]
    map_original_idx_to_new_idx = {orig_idx.item(): new_idx for new_idx, orig_idx in enumerate(original_indices_of_valid_prototypes)}  # 映射字典

    sample_filter_mask = torch.zeros_like(labels, dtype=torch.bool, device=device)  # 样本过滤掩码，形状 [batch_size]
    mapped_labels_for_loss = []  # 映射后的标签列表
    for i, label_idx_val in enumerate(labels):  # 遍历标签
        if label_idx_val.item() in map_original_idx_to_new_idx:  # 如果标签对应有效原型
            sample_filter_mask[i] = True  # 设置掩码
            mapped_labels_for_loss.append(map_original_idx_to_new_idx[label_idx_val.item()])  # 添加映射标签

    if not mapped_labels_for_loss:  # 如果没有有效样本，返回0损失
        return torch.tensor(0.0, device=device)

    valid_features = features[sample_filter_mask]  # 有效特征，形状 [num_valid_samples, feature_dim]
    mapped_labels_tensor = torch.tensor(mapped_labels_for_loss, dtype=torch.long, device=device)  # 映射标签张量，形状 [num_valid_samples]

    if valid_features.shape[0] == 0:  # 如果没有有效特征，返回0损失
        return torch.tensor(0.0, device=device)

    sim_matrix = F.cosine_similarity(valid_features.unsqueeze(1), valid_prototypes.unsqueeze(0), dim=2)  # 相似度矩阵，形状 [num_valid_samples, num_valid_prototypes]
    logits = sim_matrix / temperature  # logits，形状同上
    loss_ce_fn = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    loss_sample_to_prototype = loss_ce_fn(logits, mapped_labels_tensor)  # 样本到原型损失，标量

    loss_proto_separation = torch.tensor(0.0, device=device)  # 初始化原型分离损失
    num_valid_prototypes = valid_prototypes.shape[0]  # 有效原型数
    if num_valid_prototypes >= 2:  # 如果至少两个有效原型
        normalized_valid_prototypes = F.normalize(valid_prototypes, p=2, dim=1)  # 归一化原型，形状 [num_valid, feature_dim]
        proto_sim_matrix = F.cosine_similarity(normalized_valid_prototypes.unsqueeze(1), normalized_valid_prototypes.unsqueeze(0), dim=2)  # 原型相似度矩阵，形状 [num_valid, num_valid]
        identity_mask = torch.eye(num_valid_prototypes, dtype=torch.bool, device=device)  # 单位矩阵掩码，形状 [num_valid, num_valid]
        off_diagonal_similarities = proto_sim_matrix[~identity_mask]  # 非对角线相似度，形状 [num_valid*(num_valid-1)]
        if off_diagonal_similarities.numel() > 0:  # 如果有元素
            loss_proto_separation = off_diagonal_similarities.mean()  # 平均作为分离损失，标量

    total_contrastive_loss = loss_sample_to_prototype + lambda_ps_weight * loss_proto_separation  # 总对比损失，标量
    return total_contrastive_loss  # 返回总对比损失


def train(model, train_loader, target_loader, crit, domain_crit, optimizer,  # 定义训练函数
          lambdas_domain_w, lambda_contrastive_w, temp_cl, lambda_ps_w,
          current_epoch, total_epochs, initial_bw, max_bw, warmup_epochs_bw):
    model.train()  # 设置模型为训练模式
    loss_cls_all = 0  # 初始化分类损失累加
    loss_domain_all = 0  # 初始化域损失累加
    loss_contrastive_all = 0  # 初始化对比损失累加
    total_loss_all = 0  # 初始化总损失累加
    num_train_batches = 0  # 初始化训练批次样本数

    if current_epoch < warmup_epochs_bw:  # 如果当前轮次小于预热轮数
        current_bilinear_weight = initial_bw + (max_bw - initial_bw) * (current_epoch / float(warmup_epochs_bw))  # 计算当前双线性权重，线性增加
    else:
        current_bilinear_weight = max_bw  # 否则使用最大权重
    current_bilinear_weight = min(current_bilinear_weight, max_bw)  # 确保不超过最大值

    target_iter = iter(target_loader)  # 创建目标域数据迭代器

    for source_data in train_loader:  # 遍历源域训练加载器
        # source_data: 图数据批次，包含x [batch_nodes, features], edge_index, batch [batch_size], y [batch_size, classes]
        try:
            target_data = next(target_iter)  # 获取下一个目标域批次
        except StopIteration:  # 如果迭代器结束，重置
            target_iter = iter(target_loader)
            target_data = next(target_iter)

        source_data = source_data.to(device)  # 移动到设备
        target_data = target_data.to(device)  # 移动到设备
        optimizer.zero_grad()  # 清零梯度

        _, _, _, source_features_before_interaction = model(source_data.x, source_data.edge_index, source_data.batch)  # 前向传播获取源域特征，source_features_before_interaction [batch_size, feature_dim]

        source_label_indices = torch.argmax(source_data.y.view(-1, classes), axis=1)  # 获取源域标签索引，形状 [batch_size]

        current_source_prototypes = torch.zeros(classes, source_features_before_interaction.shape[1], device=device)  # 初始化源域原型，形状 [classes, feature_dim]
        for c_idx in range(classes):  # 遍历类别
            class_mask = (source_label_indices == c_idx)  # 类别掩码，形状 [batch_size]
            if class_mask.sum() > 0:  # 如果有样本
                current_source_prototypes[c_idx] = source_features_before_interaction[class_mask].mean(dim=0)  # 计算类别均值作为原型，形状 [feature_dim]

        source_class_out, _, source_domain_out, source_features_for_contrastive = model(  # 再次前向传播，使用原型
            source_data.x, source_data.edge_index, source_data.batch,
            source_prototypes=current_source_prototypes,
            bilinear_weight=current_bilinear_weight
        )  # source_class_out [batch_size, classes], source_domain_out [batch_size], source_features_for_contrastive [batch_size, feature_dim]
        loss_cls = crit(source_class_out, source_label_indices)  # 计算分类损失，标量

        target_class_out, _, target_domain_out, target_features_for_contrastive = model(  # 目标域前向传播
            target_data.x, target_data.edge_index, target_data.batch,
            source_prototypes=current_source_prototypes,
            bilinear_weight=current_bilinear_weight
        )  # 类似源域输出

        loss_cl_source = prototype_contrastive_loss_fn(source_features_for_contrastive, source_label_indices,  # 源域对比损失
                                                       current_source_prototypes,
                                                       temp_cl, lambda_ps_w, device)  # 标量
        with torch.no_grad():  # 无梯度计算伪标签
            pseudo_labels_target = torch.argmax(target_class_out, dim=1)  # 伪标签，形状 [batch_size]
        loss_cl_target = prototype_contrastive_loss_fn(target_features_for_contrastive, pseudo_labels_target,  # 目标域对比损失
                                                       current_source_prototypes,
                                                       temp_cl, lambda_ps_w, device)  # 标量
        loss_contrastive = (loss_cl_source + loss_cl_target) / 2.0  # 平均对比损失，标量

        source_domain_labels = torch.zeros(source_data.num_graphs, dtype=torch.float, device=device)  # 源域标签，全0，形状 [batch_size]
        target_domain_labels = torch.ones(target_data.num_graphs, dtype=torch.float, device=device)  # 目标域标签，全1，形状 [batch_size]
        domain_preds = torch.cat([source_domain_out, target_domain_out])  # 域预测拼接，形状 [2*batch_size]
        domain_labels = torch.cat([source_domain_labels, target_domain_labels])  # 域标签拼接，形状 [2*batch_size]
        loss_domain = domain_crit(domain_preds, domain_labels)  # 域损失，标量

        total_loss = loss_cls + lambdas_domain_w * loss_domain + lambda_contrastive_w * loss_contrastive  # 总损失，标量
        total_loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        loss_cls_all += loss_cls.item() * source_data.num_graphs  # 累加分类损失（加权样本数）
        loss_domain_all += loss_domain.item() * domain_labels.size(0)  # 累加域损失（加权总样本数）
        loss_contrastive_all += loss_contrastive.item() * source_data.num_graphs  # 累加对比损失
        total_loss_all += total_loss.item() * source_data.num_graphs  # 累加总损失
        num_train_batches += source_data.num_graphs  # 累加样本数

    avg_loss_cls = loss_cls_all / num_train_batches if num_train_batches > 0 else 0  # 平均分类损失
    avg_loss_domain = loss_domain_all / (num_train_batches + target_loader.batch_size * len(train_loader)) if (num_train_batches + target_loader.batch_size * len(train_loader)) > 0 else 0  # 平均域损失（注意分母计算）
    avg_loss_contrastive = loss_contrastive_all / num_train_batches if num_train_batches > 0 else 0  # 平均对比损失
    avg_total_loss = total_loss_all / num_train_batches if num_train_batches > 0 else 0  # 平均总损失

    return avg_loss_cls, avg_loss_domain, avg_loss_contrastive, avg_total_loss, current_bilinear_weight  # 返回平均损失和当前权重


def evaluate(model, loader):  # 定义评估函数
    model.eval()  # 设置模型为评估模式
    predictions = []  # 初始化预测列表
    labels = []  # 初始化标签列表

    with torch.no_grad():  # 无梯度上下文
        for data in loader:  # 遍历加载器
            label_one_hot = data.y.view(-1, classes)  # one-hot标签，形状 [batch_size, classes]
            data = data.to(device)  # 移动到设备
            class_out, pred_probs, _, _ = model(data.x, data.edge_index, data.batch)  # 前向传播，pred_probs [batch_size, classes]

            pred_probs_np = pred_probs.detach().cpu().numpy()  # 预测概率转为numpy，形状 [batch_size, classes]
            predictions.append(pred_probs_np)  # 添加到列表
            labels.append(label_one_hot.cpu().numpy())  # 添加标签到列表，形状 [batch_size, classes]

    predictions = np.vstack(predictions)  # 垂直堆叠预测，形状 [total_samples, classes]
    labels = np.vstack(labels)  # 垂直堆叠标签，形状 [total_samples, classes]

    try:  # 尝试计算AUC
        AUC = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')  # 宏平均多类AUC，标量
    except ValueError as e:  # 如果错误，设置AUC为0
        print(f"Error calculating AUC: {e}. Setting AUC to 0.")
        AUC = 0.0

    predicted_classes = np.argmax(predictions, axis=-1)  # 预测类别，形状 [total_samples]
    true_classes = np.argmax(labels, axis=-1)  # 真实类别，形状 [total_samples]

    f1 = f1_score(true_classes, predicted_classes, average='macro')  # 宏平均F1分数，标量
    acc = accuracy_score(true_classes, predicted_classes)  # 准确率，标量

    return AUC, acc, f1, labels, predictions, true_classes, predicted_classes  # 返回指标和数据


def get_roc_data(labels_one_hot, pred_probs):  # 定义获取ROC数据的函数
    """计算宏平均ROC曲线的数据点和AUC值"""
    # labels_one_hot: one-hot标签，形状 [total_samples, classes]
    # pred_probs: 预测概率，形状 [total_samples, classes]
    fpr = dict()  # FPR字典
    tpr = dict()  # TPR字典
    roc_auc = dict()  # AUC字典
    n_classes = classes  # 类别数
    for i in range(n_classes):  # 遍历类别
        fpr[i], tpr[i], _ = roc_curve(labels_one_hot[:, i], pred_probs[:, i])  # 计算每个类别的ROC曲线，fpr[i] 和 tpr[i] 为数组
        roc_auc[i] = auc(fpr[i], tpr[i])  # 计算每个类别的AUC

    # 计算宏平均
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))  # 所有FPR的唯一值，形状 [num_points]
    mean_tpr = np.zeros_like(all_fpr)  # 初始化平均TPR，形状 [num_points]
    for i in range(n_classes):  # 遍历类别
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])  # 插值TPR
    mean_tpr /= n_classes  # 平均

    macro_fpr = all_fpr  # 宏FPR
    macro_tpr = mean_tpr  # 宏TPR
    macro_auc = auc(macro_fpr, macro_tpr)  # 宏AUC

    return macro_fpr, macro_tpr, macro_auc  # 返回宏ROC数据


def main():  # 主函数
    build_dataset(subjects)  # 构建数据集
    print(f'Cross Validation with {Network.__name__}')  # 打印交叉验证信息

    all_folds_results = []  # 所有折结果列表
    detailed_results_data = []  # 详细结果数据列表

    # 用于收集所有折的真实标签和预测结果，用于最终的混淆矩阵
    all_true_classes_flat = []  # 所有真实类别平铺列表
    all_predicted_classes_flat = []  # 所有预测类别平铺列表

    # 用于绘图和保存.txt
    test_roc_data_per_subject = {}  # 每个受试者的测试ROC数据

    # 用于保存到统一的txt文件
    results_for_txt_file = []  # txt文件结果列表

    domain_crit = torch.nn.BCELoss()  # 域分类损失为BCE

    for cv_n in range(subjects):  # 遍历每个受试者作为测试折
        best_val_acc = 0.0  # 最佳验证准确率初始为0
        best_val_auc = 0.0  # 最佳验证AUC初始为0
        best_val_f1 = 0.0  # 最佳验证F1初始为0
        best_epoch = 0  # 最佳轮次初始为0
        best_model_path = f'./result/best_model_cv_{cv_n}.pth'  # 最佳模型路径

        train_dataset, test_dataset = get_dataset(subjects, cv_n)  # 获取训练和测试数据集
        target_domain_dataset = test_dataset  # 目标域数据集为测试集

        batch_size = 16  # 批次大小为16
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)  # 训练加载器
        target_loader = DataLoader(target_domain_dataset, batch_size, shuffle=True, drop_last=True)  # 目标加载器
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)  # 测试加载器
        # 创建一个用于最终评估训练集的loader
        train_eval_loader = DataLoader(train_dataset, batch_size, shuffle=False)  # 训练评估加载器

        model = PR_DA(num_classes=classes).to(device)  # 初始化模型并移动到设备
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adam优化器，学习率1e-4
        crit = torch.nn.CrossEntropyLoss()  # 交叉熵损失

        print(f"\n--- Cross-Validation Fold: Subject {cv_n} as Test ---")  # 打印当前折信息
        for epoch in range(epochs):  # 遍历轮次
            if len(train_loader) == 0 or len(target_loader) == 0:  # 如果加载器为空，跳过
                print(f"Skipping epoch {epoch + 1} for CV {cv_n} due to empty DataLoader.")
                continue

            avg_loss_cls, avg_loss_domain, avg_loss_contrastive, avg_total_loss, current_bw = train(  # 调用训练函数
                model, train_loader, target_loader, crit, domain_crit, optimizer,
                lambda_domain_weight, lambda_contrastive_weight, temperature_cl,
                lambda_proto_sep_weight,
                epoch, epochs,
                initial_bilinear_weight, max_bilinear_weight, bilinear_warmup_epochs
            )  # 获取平均损失

            # 在每个epoch后都在测试集上评估
            val_AUC, val_acc, val_f1, _, _, _, _ = evaluate(model, test_loader)  # 评估验证集（这里用测试集作为验证）

            if val_acc > best_val_acc:  # 如果当前准确率更好
                best_val_acc = val_acc  # 更新最佳准确率
                best_val_auc = val_AUC  # 更新最佳AUC
                best_val_f1 = val_f1  # 更新最佳F1
                best_epoch = epoch + 1  # 更新最佳轮次
                # 保存最佳模型
                torch.save(model.state_dict(), best_model_path)  # 保存模型状态

            print(  # 打印当前轮次信息
                f'CV{cv_n:01d}, EP{epoch + 1:03d}, BW:{current_bw:.3f}, Ls_cls:{avg_loss_cls:.4f}, Ls_dom:{avg_loss_domain:.4f}, Ls_cl:{avg_loss_contrastive:.4f}, Ls_tot:{avg_total_loss:.4f} | Val_AUC:{val_AUC:.4f}, Val_acc:{val_acc:.4f}, Val_F1:{val_f1:.4f} | Best (EP{best_epoch}): Acc={best_val_acc:.4f}, AUC={best_val_auc:.4f}, F1={best_val_f1:.4f}'
            )

        # ===== 一个被试的训练循环结束，加载最佳模型进行最终评估和数据保存 =====
        print(f"--- Evaluating best model for Subject {cv_n} (from epoch {best_epoch}) ---")  # 打印评估信息
        # 加载最佳模型
        final_model = PR_DA(num_classes=classes).to(device)  # 新模型实例
        final_model.load_state_dict(torch.load(best_model_path))  # 加载状态

        # 1. 在测试集上评估
        test_auc_val, test_acc_val, test_f1_val, test_labels, test_probs, test_true_cls, test_pred_cls = evaluate(
            final_model, test_loader)  # 评估测试集，test_labels [total_test, classes], test_probs [total_test, classes], test_true_cls [total_test]
        test_fpr, test_tpr, test_roc_auc = get_roc_data(test_labels, test_probs)  # 获取测试ROC数据

        # 2. 在训练集上评估
        train_auc_val, _, _, train_labels, train_probs, _, _ = evaluate(final_model, train_eval_loader)  # 评估训练集
        train_fpr, train_tpr, train_roc_auc = get_roc_data(train_labels, train_probs)  # 获取训练ROC数据

        # 收集该被试的结果用于最终绘图和统计
        all_folds_results.append([best_val_acc, best_val_auc, best_val_f1])  # 添加折结果
        detailed_results_data.append([cv_n, best_epoch, best_val_acc, best_val_auc, best_val_f1])  # 添加详细结果
        all_true_classes_flat.extend(test_true_cls)  # 扩展真实类别列表
        all_predicted_classes_flat.extend(test_pred_cls)  # 扩展预测类别列表
        test_roc_data_per_subject[cv_n] = (test_fpr, test_tpr, test_roc_auc)  # 保存ROC数据

        # 准备要写入txt文件的数据
        subject_data_str = f"""--- Subject {cv_n} ---
[TEST_AUC]
{test_roc_auc:.6f}
[TEST_ROC_FPR]
{','.join(map(str, test_fpr))}
[TEST_ROC_TPR]
{','.join(map(str, test_tpr))}
[TRAIN_AUC]
{train_roc_auc:.6f}
[TRAIN_ROC_FPR]
{','.join(map(str, train_fpr))}
[TRAIN_ROC_TPR]
{','.join(map(str, train_tpr))}
"""  # 格式化字符串
        results_for_txt_file.append(subject_data_str)  # 添加到列表

        # 实时保存每个被试的详细指标
        current_df = pd.DataFrame(detailed_results_data,
                                  columns=['Subject', 'Best_Epoch', 'Best_Acc', 'Best_AUC', 'Best_F1'])  # 创建DataFrame
        current_df.to_csv(f'./result/{Network.__name__}_Detail_CV_{version - 1:.0f}.csv', index=False)  # 保存CSV

        # 删除临时的模型文件
        os.remove(best_model_path)  # 删除模型文件

    # ===================================================================
    # 所有交叉验证循环结束，开始统一绘图和保存
    # ===================================================================

    # 1. 将所有数据写入单个txt文件
    with open(f'./result/all_subjects_roc_auc_data.txt', 'w') as f:  # 打开文件
        f.write('\n'.join(results_for_txt_file))  # 写入所有字符串
    print("\nAll subjects' ROC and AUC data have been saved to 'all_subjects_roc_auc_data.txt'")  # 打印保存信息

    # 2. 绘制并保存所有个体的测试集ROC曲线
    plt.figure(figsize=(10, 8))  # 创建画布
    colors = plt.cm.get_cmap('tab20', subjects)  # 获取颜色映射
    for i in range(subjects):  # 遍历受试者
        fpr, tpr, roc_auc_val = test_roc_data_per_subject[i]  # 获取数据
        plt.plot(fpr, tpr, color=colors(i), lw=2,  # 绘制曲线
                 label=f'Subject {i} (AUC = {roc_auc_val:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 绘制随机线
    plt.xlim([0.0, 1.0])  # X轴范围
    plt.ylim([0.0, 1.05])  # Y轴范围
    plt.xlabel('False Positive Rate')  # X标签
    plt.ylabel('True Positive Rate')  # Y标签
    plt.title('Macro-Average ROC Curve on Test Set for Each Subject')  # 标题
    plt.legend(loc="lower right", fontsize='small')  # 图例
    plt.grid(True)  # 网格
    plt.savefig(f'./result/ROC_Curves_All_Subjects_Test_V{version - 1}.png')  # 保存图像
    plt.show()  # 显示图像

    # 3. 绘制并保存整体混淆矩阵
    cm = confusion_matrix(all_true_classes_flat, all_predicted_classes_flat)  # 计算混淆矩阵，形状 [classes, classes]
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 归一化百分比，形状 [classes, classes]
    colors = ["#EAF2F8", "#2874A6"]  # 颜色列表
    cmap_name = 'custom_blue'  # 颜色映射名称
    cm_custom = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)  # 创建自定义颜色映射

    plt.figure(figsize=(8, 7))  # 创建画布
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cm_custom,  # 绘制热图
                xticklabels=range(classes), yticklabels=range(classes))
    plt.title('Overall Confusion Matrix on Test Set (Normalized %)')  # 标题
    plt.ylabel('True Label')  # Y标签
    plt.xlabel('Predicted Label')  # X标签
    plt.savefig(f'./result/Confusion_Matrix_V{version - 1}.png')  # 保存图像
    plt.show()  # 显示图像

    # 4. 打印最终的统计结果
    print("\n=== Final Results ===")  # 打印标题
    results_np = np.array(all_folds_results)  # 转为numpy数组，形状 [subjects, 3]
    mean_acc = np.mean(results_np[:, 0])  # 平均准确率
    std_acc = np.std(results_np[:, 0])  # 准确率标准差
    mean_auc = np.mean(results_np[:, 1])  # 平均AUC
    std_auc = np.std(results_np[:, 1])  # AUC标准差
    mean_f1 = np.mean(results_np[:, 2])  # 平均F1
    std_f1 = np.std(results_np[:, 2])  # F1标准差

    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")  # 打印平均准确率
    print(f"Mean AUC:      {mean_auc:.4f} ± {std_auc:.4f}")  # 打印平均AUC
    print(f"Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")  # 打印平均F1

    print("\n--- Individual Subject Results ---")  # 打印个体结果标题
    for i, res in enumerate(detailed_results_data):  # 遍历详细结果
        subject_id, epoch, acc, auc, f1_val = res  # 解包
        print(f"Subject {subject_id:02d} (Best Epoch {epoch}): Acc={acc:.4f}, AUC={auc:.4f}, F1={f1_val:.4f}")  # 打印

    summary_df = pd.DataFrame({  # 创建总结DataFrame
        'Metric': ['Mean Accuracy', 'Std Accuracy', 'Mean AUC', 'Std AUC', 'Mean F1-Score', 'Std F1-Score'],
        'Value': [mean_acc, std_acc, mean_auc, std_auc, mean_f1, std_f1]
    })
    summary_df.to_csv(f'./result/{Network.__name__}_Summary_{version - 1:.0f}.csv', index=False)  # 保存CSV


if __name__ == '__main__':  # 如果直接运行
    main()  # 调用主函数