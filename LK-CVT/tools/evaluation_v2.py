# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.getcwd())

import copy
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score
import matplotlib.pyplot as plt
from numpy import mean
from tqdm import tqdm
from terminaltables import AsciiTable

import torch
# import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import time
import csv

from utils.dataloader import Mydataset, collate
from utils.train_utils import file2dict
from models.build import BuildNet
from core.evaluations import evaluate
from utils.inference import init_model


# ==================================================
#               新增：辅助函数
# ==================================================
def build_dataset_from_dir(test_dir, class_to_label_map):
    """
    从目录结构动态生成 test_datas 列表。
    """
    test_datas = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    print(f"\n正在从测试目录扫描图片: {test_dir}")

    if not os.path.isdir(test_dir):
        print(f"错误: 指定的测试目录不存在: {test_dir}")
        return []

    # 遍历您定义的 CLASS_TO_LABEL_MAP
    for class_name, label_index in class_to_label_map.items():
        class_dir = os.path.join(test_dir, class_name)

        if not os.path.isdir(class_dir):
            print(f"警告: 在 {test_dir} 中未找到子目录 '{class_name}'。将跳过此类别。")
            continue

        num_found = 0
        # 遍历该类别子目录中的所有文件
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(supported_extensions):
                image_path = os.path.join(class_dir, filename)
                # 构造与 test.txt 文件行格式一致的字符串
                test_datas.append(f"{image_path} {label_index}\n")
                num_found += 1

        print(f"  - 在 '{class_name}' (标签 {label_index}) 中找到 {num_found} 张图片。")

    if not test_datas:
        print(f"错误: 在 {test_dir} 及其子目录中没有找到任何支持的图片文件。")

    print(f"总共找到 {len(test_datas)} 张测试图片。")
    return test_datas


# ==================================================


def get_metrics_output(eval_results, metrics_output, classes_names, indexs, APs):
    f = open(metrics_output, 'a', newline='')
    writer = csv.writer(f)

    """
    输出并保存Accuracy、Precision、Recall、F1 Score、Confusion matrix结果
    """
    p_r_f1 = [['Classes', 'Precision', 'Recall', 'F1 Score', 'Average Precision']]
    for i in range(len(classes_names)):
        data = []
        data.append(classes_names[i])
        data.append('{:.2f}'.format(eval_results.get('precision')[i]))
        data.append('{:.2f}'.format(eval_results.get('recall')[i]))
        data.append('{:.2f}'.format(eval_results.get('f1_score')[i]))
        data.append('{:.2f}'.format(APs[i] * 100))
        p_r_f1.append(data)
    TITLE = 'Classes Results'
    TABLE_DATA_1 = tuple(p_r_f1)
    table_instance = AsciiTable(TABLE_DATA_1, TITLE)
    # table_instance.justify_columns[2] = 'right'
    print()
    print(table_instance.table)
    writer.writerows(TABLE_DATA_1)
    writer.writerow([])
    print()

    TITLE = 'Total Results'
    TABLE_DATA_2 = (
        ('Top-1 Acc', 'Top-5 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
        ('{:.2f}'.format(eval_results.get('accuracy_top-1', 0.0)),
         '{:.2f}'.format(eval_results.get('accuracy_top-5', 100.0)),
         '{:.2f}'.format(mean(eval_results.get('precision', 0.0))),
         '{:.2f}'.format(mean(eval_results.get('recall', 0.0))),
         '{:.2f}'.format(mean(eval_results.get('f1_score', 0.0)))),
    )
    table_instance = AsciiTable(TABLE_DATA_2, TITLE)
    # table_instance.justify_columns[2] = 'right'
    print(table_instance.table)
    writer.writerows(TABLE_DATA_2)
    writer.writerow([])
    print()

    writer_list = []
    writer_list.append([' '] + [str(c) for c in classes_names])
    for i in range(len(eval_results.get('confusion'))):
        writer_list.append([classes_names[i]] + [str(x) for x in eval_results.get('confusion')[i]])
    TITLE = 'Confusion Matrix'
    TABLE_DATA_3 = tuple(writer_list)
    table_instance = AsciiTable(TABLE_DATA_3, TITLE)
    print(table_instance.table)
    writer.writerows(TABLE_DATA_3)
    print()


def get_prediction_output(preds, targets, image_paths, classes_names, indexs, prediction_output):
    nums = len(preds)
    f = open(prediction_output, 'a', newline='')
    writer = csv.writer(f)

    results = [['File', 'Pre_label', 'True_label', 'Success']]
    results[0].extend(classes_names)

    for i in range(nums):
        temp = [image_paths[i]]

        pred_idx = torch.argmax(preds[i]).item()
        true_idx = targets[i].item()

        pred_label = classes_names[pred_idx]
        true_label = classes_names[true_idx]

        success = True if pred_label == true_label else False
        class_score = preds[i].tolist()
        temp.extend([pred_label, true_label, success])
        temp.extend(class_score)
        results.append(temp)

    writer.writerows(results)


def plot_ROC_curve(preds, targets, classes_names, savedir):
    rows = len(targets)
    cols = len(preds[0])
    ROC_output = os.path.join(savedir, 'ROC')
    PR_output = os.path.join(savedir, 'P-R')
    os.makedirs(ROC_output, exist_ok=True)
    os.makedirs(PR_output, exist_ok=True)
    APs = []

    assert cols == len(classes_names), "模型输出的类别数与 classes_names 数量不匹配"

    for j in range(cols):
        gt, pre, pre_score = [], [], []
        current_class_name = classes_names[j]

        for i in range(rows):
            if targets[i].item() == j:
                gt.append(1)
            else:
                gt.append(0)

            if torch.argmax(preds[i]).item() == j:
                pre.append(1)
            else:
                pre.append(0)

            pre_score.append(preds[i][j].item())

        # ROC
        ROC_csv_path = os.path.join(ROC_output, current_class_name + '.csv')
        ROC_img_path = os.path.join(ROC_output, current_class_name + '.png')
        ROC_f = open(ROC_csv_path, 'a', newline='')
        ROC_writer = csv.writer(ROC_f)
        ROC_results = []

        FPR, TPR, threshold = roc_curve(targets.tolist(), pre_score, pos_label=j)

        AUC = auc(FPR, TPR)

        ROC_results.append(['AUC', AUC])
        ROC_results.append(['FPR'] + FPR.tolist())
        ROC_results.append(['TPR'] + TPR.tolist())
        ROC_results.append(['Threshold'] + threshold.tolist())
        ROC_writer.writerows(ROC_results)

        plt.figure()
        plt.title(current_class_name + ' ROC CURVE (AUC={:.2f})'.format(AUC))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.plot(FPR, TPR, color='g')
        plt.plot([0, 1], [0, 1], color='m', linestyle='--')
        plt.savefig(ROC_img_path)

        # AP (gt为{0,1})
        AP = average_precision_score(gt, pre_score)
        APs.append(AP)

        # P-R
        PR_csv_path = os.path.join(PR_output, current_class_name + '.csv')
        PR_img_path = os.path.join(PR_output, current_class_name + '.png')
        PR_f = open(PR_csv_path, 'a', newline='')
        PR_writer = csv.writer(PR_f)
        PR_results = []

        PRECISION, RECALL, thresholds = precision_recall_curve(targets.tolist(), pre_score, pos_label=j)

        PR_results.append(['RECALL'] + RECALL.tolist())
        PR_results.append(['PRECISION'] + PRECISION.tolist())
        PR_results.append(['Threshold'] + thresholds.tolist())
        PR_writer.writerows(PR_results)

        plt.figure()
        plt.title(current_class_name + ' P-R CURVE (AP={:.2f})'.format(AP))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.plot(RECALL, PRECISION, color='g')
        plt.savefig(PR_img_path)

    return APs


def main():
    # ==================================================
    #               用户自定义变量
    # ==================================================
    # 1. 配置文件路径
    config_file_path = '/home/UserData/les/Awesome-Backbones-main/datas/train_output/测试集1-增强_诊断二分类/tinyvit_21m.py'

    # 2. 结果保存目录
    custom_save_dir = '/home/UserData/les/Awesome-Backbones-main/datas/eval_results/测试集1增强模型_诊断二分类_德阳'

    # 3. GPU ID
    gpu_id = 0

    # 4.1 [关键修改] 数据加载映射：文件夹名 -> 标签 ID
    # 这里定义从哪些文件夹读取图片，以及它们对应的真实标签是 0 还是 1
    CLASS_TO_LABEL_MAP = {
        # "四腔心 - -100张": 0,  # 标签 0
        # "主动脉弓-100张": 1,  # 标签 1
        # "动脉导管弓--100张": 1,  # 标签 1
        # "右室流出道-100张": 1,  # 标签 1
        # "左室流出道-100张": 1,  # 标签 1
        # "腹部横切面-100张": 1  # 标签 1
        "正常":0,
        "异常":1
    }

    # 4.2 [新增必填] 模型类别名称：索引 ID -> 类别名
    # 列表长度必须等于模型的 num_classes (这里是 2)
    # 第 0 个元素对应 Label 0 的名字，第 1 个元素对应 Label 1 的名字
    MODEL_CLASS_NAMES = ["正常", "异常"]

    # 5. 测试集根目录
    test_dir_path = '/home/UserData/les/增强结果/2025.11.20测试集2-按质量好vs质量差分类-16538张/质量差-9441张'

    # 6. 权重文件路径
    checkpoint_file_path = '/home/UserData/les/Awesome-Backbones-main/datas/train_output/测试集1-增强_诊断二分类/Last_Epoch100.pth'
    # ==================================================

    print(f"正在加载配置文件: {config_file_path}")
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(config_file_path)

    print(f"将使用本地评估权重: {checkpoint_file_path}")
    data_cfg['test']['ckpt'] = checkpoint_file_path

    # 修改 val_pipeline 以用于评估 (添加 'gt_label')
    print("正在修改 val_pipeline 以用于评估 (添加 'gt_label')...")
    found_collect = False
    for step in val_pipeline:
        if step.get('type') == 'Collect':
            if 'gt_label' not in step['keys']:
                step['keys'].append('gt_label')
            found_collect = True
            break
    if not found_collect:
        print("警告: 在 val_pipeline 中未找到 'Collect' 步骤。")

    # 创建保存目录
    save_dir = custom_save_dir
    metrics_output = os.path.join(save_dir, 'metrics_output.csv')
    prediction_output = os.path.join(save_dir, 'prediction_results.csv')
    label_map_output = os.path.join(save_dir, 'label_map.txt')
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------------
    # [逻辑修正] 不再根据文件夹数量生成类别名，而是使用 MODEL_CLASS_NAMES
    # -------------------------------------------------------
    classes_names = MODEL_CLASS_NAMES
    # 生成索引列表 [0, 1]
    indexs = list(range(len(classes_names)))

    print("\n正在使用模型类别定义:")
    for idx, name in enumerate(classes_names):
        print(f"  Label {idx} -> {name}")

    # 加载数据 (使用 CLASS_TO_LABEL_MAP 查找文件)
    test_datas = build_dataset_from_dir(test_dir_path, CLASS_TO_LABEL_MAP)

    # 保存简单的 Label Map 说明
    with open(label_map_output, 'w', encoding='utf-8') as f:
        f.write("Model Output Class Map\n")
        for idx, name in enumerate(classes_names):
            f.write(f"{idx}: {name}\n")

    # 生成模型、加载权重
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = BuildNet(model_cfg)
    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[gpu_id])
    model = init_model(model, data_cfg, device=device, mode='eval')

    # 制作测试集
    test_dataset = Mydataset(test_datas, val_pipeline)
    batch_size = data_cfg.get('batch_size', 1)
    if batch_size <= 0: batch_size = 1
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                             num_workers=data_cfg.get('num_workers', 0), pin_memory=True, collate_fn=collate)

    print("\n开始模型评估...")
    with torch.no_grad():
        preds, targets, image_paths = [], [], []
        if len(test_datas) == 0:
            print("错误：测试数据为空。")
            return

        # 进度条逻辑
        total_batches = max(1, len(test_datas) // batch_size)
        if len(test_datas) > batch_size and len(test_datas) % batch_size != 0:
            total_batches += 1

        with tqdm(total=total_batches) as pbar:
            for _, batch in enumerate(test_loader):
                images, target, image_path = batch
                outputs = model(images.to(device), return_loss=False)
                preds.append(outputs)
                targets.append(target.to(device))
                image_paths.extend(image_path)
                pbar.update(1)

    print("评估完成，正在计算指标...")
    if not preds: return

    all_preds = torch.cat(preds)
    all_targets = torch.cat(targets)

    # 检查维度匹配
    num_classes_model = all_preds.shape[1]
    num_classes_map = len(classes_names)

    if num_classes_model != num_classes_map:
        print(f"\n严重错误：")
        print(f"  模型输出维度: {num_classes_model} (由 config 定义)")
        print(f"  脚本预设维度: {num_classes_map} (由 MODEL_CLASS_NAMES 定义)")
        print("  请确保 MODEL_CLASS_NAMES 的长度等于模型的分类数！")
        return

    # 计算指标
    eval_results = evaluate(all_preds, all_targets, data_cfg.get('test').get('metrics'),
                            data_cfg.get('test').get('metric_options'))

    # 绘图和输出
    # 注意：这里的 classes_names 只有两个元素，这正是 plot_ROC_curve 需要的
    APs = plot_ROC_curve(all_preds, all_targets, classes_names, save_dir)
    get_metrics_output(eval_results, metrics_output, classes_names, indexs, APs)
    get_prediction_output(all_preds, all_targets, image_paths, classes_names, indexs, prediction_output)

    print(f"\n所有评估结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()