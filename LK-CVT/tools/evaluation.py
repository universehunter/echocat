import os
import sys

sys.path.insert(0, os.getcwd())
import argparse
import copy
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, \
    accuracy_score
import matplotlib.pyplot as plt
from numpy import mean
from tqdm import tqdm
from terminaltables import AsciiTable
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import time
import csv

# 自定义模块引入
from utils.dataloader import Mydataset, collate
from utils.train_utils import get_info, file2dict
from models.build import BuildNet
from core.evaluations import evaluate
from utils.inference import init_model

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_metrics_output(eval_results, metrics_output, classes_names, indexs, APs, Accs):
    f = open(metrics_output, 'a', newline='')
    writer = csv.writer(f)

    # 表头增加 'Accuracy'
    p_r_f1 = [['Classes', 'Precision', 'Recall', 'F1 Score', 'Average Precision', 'Accuracy']]

    for i in range(len(classes_names)):
        data = []
        data.append(classes_names[i])
        data.append('{:.2f}'.format(eval_results.get('precision')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('recall')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('f1_score')[indexs[i]]))
        data.append('{:.2f}'.format(APs[indexs[i]] * 100))

        # 添加 Accuracy 数据 (转为百分比)
        data.append('{:.2f}'.format(Accs[indexs[i]] * 100))

        p_r_f1.append(data)

    TITLE = 'Classes Results'
    TABLE_DATA_1 = tuple(p_r_f1)
    table_instance = AsciiTable(TABLE_DATA_1, TITLE)
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
        pred_label = classes_names[indexs[torch.argmax(preds[i]).item()]]
        true_label = classes_names[indexs[targets[i].item()]]
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
    if not os.path.exists(ROC_output): os.makedirs(ROC_output)
    if not os.path.exists(PR_output): os.makedirs(PR_output)

    APs = []
    Accs = []  # 初始化 Accuracy 列表

    for j in range(cols):
        gt, pre, pre_score = [], [], []
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

        ROC_csv_path = os.path.join(ROC_output, classes_names[j] + '.csv')
        ROC_img_path = os.path.join(ROC_output, classes_names[j] + '.png')
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
        plt.title(classes_names[j] + ' ROC CURVE (AUC={:.2f})'.format(AUC))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.plot(FPR, TPR, color='g')
        plt.plot([0, 1], [0, 1], color='m', linestyle='--')
        plt.savefig(ROC_img_path)

        AP = average_precision_score(gt, pre_score)
        APs.append(AP)

        # 计算 One-vs-Rest Accuracy 并添加
        acc = accuracy_score(gt, pre)
        Accs.append(acc)

        PR_csv_path = os.path.join(PR_output, classes_names[j] + '.csv')
        PR_img_path = os.path.join(PR_output, classes_names[j] + '.png')
        PR_f = open(PR_csv_path, 'a', newline='')
        PR_writer = csv.writer(PR_f)
        PR_results = []
        PRECISION, RECALL, thresholds = precision_recall_curve(targets.tolist(), pre_score, pos_label=j)
        PR_results.append(['RECALL'] + RECALL.tolist())
        PR_results.append(['PRECISION'] + PRECISION.tolist())
        PR_results.append(['Threshold'] + thresholds.tolist())
        PR_writer.writerows(PR_results)
        plt.figure()
        plt.title(classes_names[j] + ' P-R CURVE (AP={:.2f})'.format(AP))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.plot(RECALL, PRECISION, color='g')
        plt.savefig(PR_img_path)

    return APs, Accs


def main():
    # ========================== 用户变量设置区域 ==========================

    # 1. 配置文件路径
    config_file_path = '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/华西诊断训练集增强-重新训练/tinyvit_21m.py'

    # 2. 模型权重文件路径 (.pth)
    checkpoint_file_path = '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/华西诊断训练集增强-重新训练/Last_Epoch100.pth'

    # 3. 类别映射文件路径
    classes_map = '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类1.15/label_map.txt'

    # 4. 测试集标注文件路径 (使用你新生成的txt)
    test_annotations = '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类1.15/质量差-9441张-labels.txt'

    # 5. GPU 设置
    gpu_id = 0
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 6. 【修改点】输出结果保存目录 (根据 input 文件名自动生成)
    # 逻辑：获取 test_annotations 的目录，取其文件名去掉后缀，拼接 "评估结果"
    input_dir = os.path.dirname(test_annotations)
    input_filename = os.path.basename(test_annotations)
    file_name_no_ext = os.path.splitext(input_filename)[0]

    # 自动组合路径： /原目录/文件名 + 评估结果
    output_dir = os.path.join(input_dir, file_name_no_ext + "评估结果")

    # ====================================================================

    # 加载配置文件
    print(f"正在加载配置文件: {config_file_path}")
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(config_file_path)

    """
    创建评估文件夹、metrics文件、混淆矩阵文件
    """
    # 使用用户指定的 output_dir
    save_dir = output_dir

    metrics_output = os.path.join(save_dir, 'metrics_output.csv')
    prediction_output = os.path.join(save_dir, 'prediction_results.csv')

    if not os.path.exists(save_dir):
        print(f"创建输出目录: {save_dir}")
        os.makedirs(save_dir)
    else:
        print(f"输出目录已存在: {save_dir}")

    """
    获取类别名以及对应索引、获取标注文件
    """
    classes_names, indexs = get_info(classes_map)
    with open(test_annotations, encoding='utf-8') as f:
        test_datas = f.readlines()

    """
    生成模型、加载权重
    """
    device = torch.device(device_name)

    print("正在构建模型...")
    model = BuildNet(model_cfg)

    # 手动加载权重文件
    print(f"正在加载权重文件: {checkpoint_file_path}")
    checkpoint = torch.load(checkpoint_file_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 去除 module. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"权重加载完成: {msg}")

    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[gpu_id])

    # 手动设置 eval 模式
    model.to(device)
    model.eval()

    """
    制作测试集并喂入Dataloader
    """
    val_pipeline = copy.deepcopy(train_pipeline)
    test_dataset = Mydataset(test_datas, val_pipeline)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'),
                             num_workers=data_cfg.get('num_workers'), pin_memory=True, collate_fn=collate)

    """
    开始推理
    """
    print("开始推理...")
    with torch.no_grad():
        preds, targets, image_paths = [], [], []
        with tqdm(total=len(test_loader)) as pbar:
            for _, batch in enumerate(test_loader):
                images, target, image_path = batch
                outputs = model(images.to(device), return_loss=False)
                preds.append(outputs)
                targets.append(target.to(device))
                image_paths.extend(image_path)
                pbar.update(1)

    eval_results = evaluate(torch.cat(preds), torch.cat(targets), data_cfg.get('test').get('metrics'),
                            data_cfg.get('test').get('metric_options'))

    # 接收 APs 和 Accs
    APs, Accs = plot_ROC_curve(torch.cat(preds), torch.cat(targets), classes_names, save_dir)

    # 传递 Accs
    get_metrics_output(eval_results, metrics_output, classes_names, indexs, APs, Accs)

    get_prediction_output(torch.cat(preds), torch.cat(targets), image_paths, classes_names, indexs, prediction_output)
    print(f"评估完成！结果已保存至: {save_dir}")


if __name__ == "__main__":
    main()