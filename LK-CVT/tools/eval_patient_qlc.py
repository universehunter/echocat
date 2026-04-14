# -*- coding: utf-8 -*-
import os
import sys
import csv
import copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from numpy import mean
from tqdm import tqdm
from terminaltables import AsciiTable
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

# 解决路径问题
sys.path.insert(0, os.getcwd())

# 自定义模块引入
from utils.dataloader import Mydataset, collate
from utils.train_utils import get_info, file2dict
from models.build import BuildNet
from core.evaluations import evaluate

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 全流程评估任务配置 =================

# 切面分类模型配置
PLANE_CLASSIFIER_CONFIG = '/home/UserData/les/Awesome-Backbones-main/datas/增强切面/华西增强切面/tinyvit_21m.py'
PLANE_CLASSIFIER_CHECKPOINT = '/home/UserData/les/Awesome-Backbones-main/datas/增强切面/华西增强切面/Val_Epoch139-Acc97.390.pth'
PLANE_LABEL_MAP = '/home/UserData/les/Awesome-Backbones-main/datas/胎儿诊断全流程/qm_class_map.txt'

# 疾病诊断模型配置
DIAGNOSIS_CONFIG = '/home/UserData/les/Awesome-Backbones-main/datas/train_output/测试集1-增强_诊断二分类/tinyvit_21m.py'
DIAGNOSIS_CHECKPOINT = '/home/UserData/zyh/华西新/Awesome-Backbones-main/logs/TinyViT/2026-01-29-16-29-02/Train_Epoch147-Loss0.108.pth'
DIAGNOSIS_LABEL_MAP = '/home/UserData/les/Awesome-Backbones-main/datas/胎儿分类-12.13/class_map.txt'

# 四腔心类别索引（根据你的切面分类模型设定）
FOUR_CHAMBER_CLASS_INDEX = 0  # 假设四腔心在切面分类模型中是第0类

# 统一输出目录
BASE_OUTPUT_DIR = '/home/UserData/les/Awesome-Backbones-main/datas/胎儿诊断全流程/全流程评估结果3-3'

BATCH_TASKS = [
    # --- 任务 1: 德阳增强 ---
    # {
    #     'task_name': '德阳增强',
    #     'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿诊断全流程/德阳增强.txt',
    #     'id_rules': {
    #         '异常组-52例': 3,
    #         '正常组-132例（含32例点状强回声）': 2,
    #     }
    # },
    #
    # # --- 任务 2: 德阳未增强 ---
    # {
    #     'task_name': '德阳未增强',
    #     'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿诊断全流程/德阳未增强.txt',
    #     'id_rules': {
    #         '异常组-52例': 3,
    #         '正常组-132例（含32例点状强回声）': 2,
    #     }
    # },
    #
    # # --- 任务 3: 省妇保增强 ---
    {
        'task_name': '省妇保增强',
        'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿诊断全流程/省妇保增强.txt',
        'id_rules': {
            '2025.5.27叶璐核实完成切面分类标注-20255.15潘仁咪-2025.3.25再次标注-省妇保正常组--375例': 2,
            '2025.6.2叶璐完成切面分类及四腔心图片诊断分类标注-2025.5.27潘仁咪-2025.3.25再次标注完成-2月刘杨陈琳返回-省妇保异常组-375例': 2,
        }
    },
    
    # # --- 任务 4: 省妇保未增强 ---
    # {
    #     'task_name': '省妇保未增强',
    #     'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿诊断全流程/省妇保未增强.txt',
    #     'id_rules': {
    #         '2025.5.27叶璐核实完成切面分类标注-20255.15潘仁咪-2025.3.25再次标注-省妇保正常组--375例': 2,
    #         '2025.6.2叶璐完成切面分类及四腔心图片诊断分类标注-2025.5.27潘仁咪-2025.3.25再次标注完成-2月刘杨陈琳返回-省妇保异常组-375例': 2,
    #     }
    # },
    #
    # --- 任务 5: 测试集2增强 ---
    # {
    #     'task_name': '测试集2增强',
    #     'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿诊断全流程/测试集2增强.txt',
    #     'id_rules': {
    #         '2025.3.31再次标注测试集2正常组-全部预处理后-40例': (2, 0),
    #         '2025.4.14再次标注测试集2异常组-四腔心图片标注正常与异常-全部预处理后-39例': (2, 1),
    #     }
    # },
    #
    # # --- 任务 6: 测试集2未增强 ---
    # {
    #     'task_name': '测试集2未增强',
    #     'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿诊断全流程/测试集2未增强.txt',
    #     'id_rules': {
    #         '2025.3.31再次标注测试集2正常组-全部预处理后-40例': (2, 0),
    #         '2025.4.14再次标注测试集2异常组-四腔心图片标注正常与异常-全部预处理后-39例': (2, 1),
    #     }
    # },
    # {
    #     'task_name': '万源未增强',
    #     'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿诊断全流程/万源未增强.txt',
    #     'id_rules': {
    #         '2025.3-4月标注-万源正常组-已标注121例-2025.4.19': (2, 0),
    #         '2025.3.26重新标注-万源市中心医院异常组-专家返回后-11例': (1, 1)
    #     }
    # },
    # --- 任务 6: 测试集2未增强 ---
    # {
    #     'task_name': '万源增强',
    #     'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿诊断全流程/万源增强.txt',
    #     'id_rules': {
    #         '2025.3-4月标注-万源正常组-已标注121例-2025.4.19': (2, 0),
    #         '2025.3.26重新标注-万源市中心医院异常组-专家返回后-11例': (1, 1)
    #     }
    # }
]


# ================= 2. 核心逻辑修改区域 =================

def load_image_paths_only(file_path):
    """
    加载只有图片路径的标签文件
    每行只有一个图片路径，没有标签
    """
    print(f"加载图片路径文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    image_paths = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # 假设每行只有一个图片路径
        image_paths.append(line)

    print(f"共加载 {len(image_paths)} 个图片路径")
    return image_paths


def convert_paths_to_dummy_labels(image_paths, dummy_label=0):
    """
    将只有路径的列表转换为带伪标签的格式
    返回格式: ["路径 伪标签"]
    """
    return [f"{path} {dummy_label}" for path in image_paths]


def get_id_and_gt_by_rules(path, rules):
    """
    根据规则提取病人ID。
    支持两种配置格式：
    1. '关键词': 偏移量          -> 自动根据关键词里的"正常/异常"猜标签
    2. '关键词': (偏移量, 标签)  -> [推荐] 强制指定标签，解决混合关键词问题
    """
    path = os.path.normpath(path)
    parts = path.split(os.sep)

    found_id = None
    rule_label = None

    for keyword, value in rules.items():
        # === 解析配置 ===
        if isinstance(value, tuple) or isinstance(value, list):
            # 新写法: (偏移量, 强制标签)
            offset = value[0]
            fixed_label = value[1]
        else:
            # 旧写法: 只有偏移量
            offset = value
            fixed_label = None

        # === 匹配关键词 ===
        # 查找关键词在路径中的位置
        match_index = -1
        for i, part in enumerate(parts):
            if keyword in part:
                match_index = i
                break

        if match_index != -1:
            target_index = match_index + offset
            if target_index < len(parts):
                found_id = parts[target_index]

                # === 标签判定逻辑 ===
                if fixed_label is not None:
                    # 优先级最高：使用了 (offset, label) 格式，直接用指定标签
                    rule_label = fixed_label
                else:
                    # 优先级低：自动推断 (容易被混合文件名误导)
                    if '正常' in keyword and '异常' not in keyword:
                        rule_label = 0
                    elif '异常' in keyword and '正常' not in keyword:
                        rule_label = 1
                    elif '正常' in keyword and '异常' in keyword:
                        # 如果同时包含，尝试优先匹配"异常组"这个词
                        if '异常组' in keyword:
                            rule_label = 1
                        elif '正常组' in keyword:
                            rule_label = 0
                        else:
                            print(f"[警告] 文件夹 '{keyword}' 同时包含正常和异常，且未指定强制标签，可能会误判！")
                            # 默认兜底逻辑（以前的逻辑），这里可以改
                            if '异常' in keyword:
                                rule_label = 1
                            else:
                                rule_label = 0
                    elif '正常' in keyword:
                        rule_label = 0
                    elif '异常' in keyword:
                        rule_label = 1

                return found_id, rule_label

    return None, None


def filter_four_chamber_images(plane_model, data_loader, device, four_chamber_index):
    """
    使用切面分类模型筛选四腔心图片
    """
    print(">>> 使用切面分类模型筛选四腔心图片...")
    plane_model.eval()

    four_chamber_paths = []
    four_chamber_targets = []

    with torch.no_grad():
        for _, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, target, path = batch
            outputs = plane_model(images.to(device), return_loss=False)
            outputs = F.softmax(outputs, dim=1)

            # 获取预测类别
            preds = torch.argmax(outputs, dim=1)

            # 筛选四腔心图片
            for i in range(len(preds)):
                if preds[i].item() == four_chamber_index:
                    four_chamber_paths.append(path[i])
                    # 注意：这里的target是伪标签，我们不使用它
                    # 保留伪标签只是为了保持数据结构一致
                    four_chamber_targets.append(target[i])

    print(f"筛选完成！共找到 {len(four_chamber_paths)} 张四腔心图片")
    return four_chamber_paths, four_chamber_targets


def evaluate_patient_level_by_rules(preds, targets, image_paths, classes_names, data_cfg, save_dir, rules):
    """
    使用 rules 对图片进行归类，并根据 rule key 决定最终标签
    """
    print(f"正在进行胎儿级别聚合 (Label 判定依据: 文件夹规则)...")

    patient_data = {}  # {pid: {'scores': [], 'rule_label': int}}
    missing_rule_count = 0

    # 1. 遍历所有图片
    for i, path in enumerate(image_paths):
        # 获取 ID 和 强制标签
        pid, rule_lbl = get_id_and_gt_by_rules(path, rules)

        if pid:
            if pid not in patient_data:
                patient_data[pid] = {'scores': [], 'rule_label': rule_lbl}

            patient_data[pid]['scores'].append(preds[i])

            # 校验一致性 (可选)
            if patient_data[pid]['rule_label'] is not None and rule_lbl is not None:
                if patient_data[pid]['rule_label'] != rule_lbl:
                    print(f"[警告] 病人 {pid} 存在标签冲突！")
        else:
            missing_rule_count += 1

    if missing_rule_count > 0:
        print(f"[警告] 有 {missing_rule_count} 张图片未匹配到ID规则，已跳过。")

    # 2. 生成最终数据
    final_patient_ids = []
    final_preds = []
    final_targets = []

    unknown_label_count = 0

    for pid in sorted(patient_data.keys()):
        data = patient_data[pid]

        # 检查该病人是否有四腔心图片
        if not data['scores']:
            print(f"[警告] 病人 {pid} 没有四腔心图片，跳过")
            continue

        # Mean Pooling
        scores_stack = torch.stack(data['scores'])
        avg_score = torch.mean(scores_stack, dim=0)

        # === [修改点] 直接使用规则定义的 Label ===
        gt = data['rule_label']

        if gt is None:
            # 如果规则里既没有"正常"也没有"异常"，则无法判定 (防守逻辑)
            unknown_label_count += 1
            continue

        final_patient_ids.append(pid)
        final_preds.append(avg_score)
        final_targets.append(gt)

    if unknown_label_count > 0:
        print(f"[提示] 有 {unknown_label_count} 个病人无法通过关键词(正常/异常)确定标签，已跳过。")

    if len(final_patient_ids) == 0:
        print("[错误] 有效病人数量为 0！请检查 rules 里的关键词是否包含 '正常' 或 '异常'。")
        return {}, [], [], [], [], []

    final_preds_tensor = torch.stack(final_preds)
    final_targets_tensor = torch.tensor(final_targets)

    print(f"聚合完成！共评估 {len(final_patient_ids)} 个胎儿。")

    # 打印简要分布
    n_normal = sum([1 for t in final_targets if t == 0])
    n_abnormal = sum([1 for t in final_targets if t == 1])
    print(f"  -> 正常组(0): {n_normal} 例")
    print(f"  -> 异常组(1): {n_abnormal} 例")

    # 3. 计算指标
    THRESHOLD = 0.575
    print(f"\n>>> 应用自定义阈值: {THRESHOLD} (仅针对混淆矩阵和Acc/Recall/Precision表)")
    
    # 1. 获取“异常”类别的概率 (假设第1列是异常，第0列是正常)
    # final_preds_tensor shape: [N, 2]
    abnormal_probs = final_preds_tensor[:, 1]
    
    # 2. 根据阈值生成 0 或 1 的硬标签
    # 如果 异常概率 >= 0.7，则为 1 (异常)；否则为 0 (正常)
    hard_preds = (abnormal_probs >= THRESHOLD).long()
    
    # 3. 将硬标签转回 One-Hot 格式，伪装成概率喂给 evaluate
    # 结果类似: [[1, 0], [0, 1], ...] 这样 evaluate 里的 argmax 就会完全听从我们的阈值
    thresholded_preds_tensor = torch.nn.functional.one_hot(hard_preds, num_classes=2).float()

    # 4. 计算指标 (注意：这里传入 thresholded_preds_tensor)
    # 这样打印出来的 Table、Recall、Precision 都是基于 0.7 阈值的
    eval_results = evaluate(thresholded_preds_tensor, final_targets_tensor,
                            data_cfg.get('test').get('metrics'),
                            data_cfg.get('test').get('metric_options'))

    # ================= 修改结束 =================

    # 5. 绘图 
    # 【注意】画 ROC/AUC 曲线必须用原始的概率 (final_preds_tensor)，不能用截断后的！
    # 因为 ROC 就是要看不同阈值下的表现，不需要我们手动卡阈值。
    APs, Accs = plot_ROC_curve(final_preds_tensor, final_targets_tensor, classes_names, save_dir, prefix='Patient_')

    # 【注意】这里返回的时候，你可以选择返回 截断后的 preds 用于生成 csv，或者原始的
    # 如果希望 CSV 里“Pre_label”列也体现 0.7 的效果，建议返回 thresholded_preds_tensor
    return eval_results, APs, Accs, thresholded_preds_tensor, final_targets_tensor, final_patient_ids


# ================= 3. 辅助函数区域 =================

def get_metrics_output(eval_results, metrics_output, classes_names, indexs, APs, Accs, title_prefix=""):
    f = open(metrics_output, 'a', newline='')
    writer = csv.writer(f)
    if title_prefix:
        writer.writerow([f"=== {title_prefix} Results ==="])
        print(f"\n=== {title_prefix} Results ===")

    p_r_f1 = [['Classes', 'Precision', 'Recall', 'F1 Score', 'Average Precision', 'Accuracy']]
    for i in range(len(classes_names)):
        data = []
        data.append(classes_names[i])
        data.append('{:.2f}'.format(eval_results.get('precision')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('recall')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('f1_score')[indexs[i]]))
        ap_val = APs[indexs[i]] * 100 if APs and i < len(APs) else 0.0
        acc_val = Accs[indexs[i]] * 100 if Accs and i < len(Accs) else 0.0
        data.append('{:.2f}'.format(ap_val))
        data.append('{:.2f}'.format(acc_val))
        p_r_f1.append(data)

    TITLE = f'{title_prefix} Classes Results'
    table_instance = AsciiTable(tuple(p_r_f1), TITLE)
    print(table_instance.table)
    writer.writerows(tuple(p_r_f1))
    writer.writerow([])

    TITLE = f'{title_prefix} Total Results'
    TABLE_DATA_2 = (
        ('Top-1 Acc', 'Top-5 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
        ('{:.2f}'.format(eval_results.get('accuracy_top-1', 0.0)),
         '{:.2f}'.format(eval_results.get('accuracy_top-5', 100.0)),
         '{:.2f}'.format(mean(eval_results.get('precision', 0.0))),
         '{:.2f}'.format(mean(eval_results.get('recall', 0.0))),
         '{:.2f}'.format(mean(eval_results.get('f1_score', 0.0)))),)
    table_instance = AsciiTable(TABLE_DATA_2, TITLE)
    print(table_instance.table)
    writer.writerows(TABLE_DATA_2)
    writer.writerow([])

    writer_list = []
    writer_list.append([' '] + [str(c) for c in classes_names])
    cm = eval_results.get('confusion')
    if len(cm) == len(classes_names):
        for i in range(len(cm)):
            writer_list.append([classes_names[i]] + [str(x) for x in cm[i]])
    TITLE = f'{title_prefix} Confusion Matrix'
    table_instance = AsciiTable(tuple(writer_list), TITLE)
    print(table_instance.table)
    writer.writerows(tuple(writer_list))
    print()


def get_prediction_output(preds, targets, identifiers, classes_names, indexs, prediction_output, header_id="File"):
    nums = len(preds)
    f = open(prediction_output, 'a', newline='')
    writer = csv.writer(f)
    results = [[header_id, 'Pre_label', 'True_label', 'Success']]
    results[0].extend(classes_names)

    for i in range(nums):
        temp = [identifiers[i]]
        pred_idx = torch.argmax(preds[i]).item()
        target_idx = targets[i].item()
        pred_label = classes_names[indexs[pred_idx]]
        true_label = classes_names[indexs[target_idx]]
        success = True if pred_label == true_label else False
        class_score = preds[i].tolist()
        temp.extend([pred_label, true_label, success])
        temp.extend(['{:.4f}'.format(s) for s in class_score])
        results.append(temp)
    writer.writerows(results)


def plot_ROC_curve(preds, targets, classes_names, savedir, prefix=""):
    rows = len(targets)
    cols = len(preds[0])
    ROC_output = os.path.join(savedir, 'ROC')
    PR_output = os.path.join(savedir, 'P-R')
    if not os.path.exists(ROC_output): os.makedirs(ROC_output)
    if not os.path.exists(PR_output): os.makedirs(PR_output)
    APs, Accs = [], []

    for j in range(cols):
        gt, pre, pre_score = [], [], []
        for i in range(rows):
            gt.append(1 if targets[i].item() == j else 0)
            pre.append(1 if torch.argmax(preds[i]).item() == j else 0)
            pre_score.append(preds[i][j].item())
        try:
            FPR, TPR, threshold = roc_curve(gt, pre_score)
            AUC_score = auc(FPR, TPR)
            plt.figure()
            plt.title(f'{prefix}{classes_names[j]} ROC (AUC={AUC_score:.2f})')
            plt.plot(FPR, TPR, color='darkorange', lw=2, label=f'AUC={AUC_score:.2f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(ROC_output, f"{prefix}{classes_names[j]}.png"))
            plt.close()

            PRECISION, RECALL, _ = precision_recall_curve(gt, pre_score)
            AP = average_precision_score(gt, pre_score)
            plt.figure()
            plt.title(f'{prefix}{classes_names[j]} P-R (AP={AP:.2f})')
            plt.plot(RECALL, PRECISION, color='blue', lw=2, label=f'AP={AP:.2f}')
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(PR_output, f"{prefix}{classes_names[j]}.png"))
            plt.close()
            APs.append(AP)
        except:
            APs.append(0.0)
        Accs.append(accuracy_score(gt, pre))
    return APs, Accs


def create_filtered_dataset(four_chamber_paths, four_chamber_targets):
    """
    创建筛选后的数据集，包含四腔心图片
    注意：这里不需要原始数据行，因为我们已经有了筛选后的路径和伪标签
    """
    filtered_data = []
    for path, target in zip(four_chamber_paths, four_chamber_targets):
        # 使用伪标签构建数据集格式
        filtered_data.append(f"{path} {target.item()}")

    return filtered_data


def run_single_task(task_conf):
    task_name = task_conf['task_name']
    print(f"\n{'=' * 20} 开始任务: {task_name} {'=' * 20}")

    # 配置参数
    test_annotations = task_conf['test_annotations']
    id_rules = task_conf['id_rules']

    # 使用全局模型配置
    plane_config_file = PLANE_CLASSIFIER_CONFIG
    plane_checkpoint_file = PLANE_CLASSIFIER_CHECKPOINT
    plane_label_map = PLANE_LABEL_MAP
    diagnosis_config_file = DIAGNOSIS_CONFIG
    diagnosis_checkpoint_file = DIAGNOSIS_CHECKPOINT
    diagnosis_label_map = DIAGNOSIS_LABEL_MAP

    # 生成输出目录：基础目录 + 任务名称
    output_dir = os.path.join(BASE_OUTPUT_DIR, task_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================= 加载只有路径的标签文件 =================
    print(">>> 加载图片路径文件...")
    image_paths = load_image_paths_only(test_annotations)

    if len(image_paths) == 0:
        print(f"[错误] 标签文件 {test_annotations} 没有有效数据，跳过该任务")
        return

    # 转换为带伪标签的格式，以便Mydataset可以处理
    dummy_label_data = convert_paths_to_dummy_labels(image_paths, dummy_label=0)

    # ================= 第一阶段：切面分类筛选 =================
    print(">>> 第一阶段：加载切面分类模型...")
    plane_model_cfg, plane_train_pipeline, _, plane_data_cfg, _, _ = file2dict(plane_config_file)
    plane_classes_names, plane_indexs = get_info(plane_label_map)

    # 加载切面分类模型
    plane_model = BuildNet(copy.deepcopy(plane_model_cfg))
    plane_checkpoint = torch.load(plane_checkpoint_file, map_location='cpu')
    plane_state_dict = plane_checkpoint['state_dict'] if 'state_dict' in plane_checkpoint else plane_checkpoint
    new_plane_state_dict = {k.replace('module.', ''): v for k, v in plane_state_dict.items()}
    plane_model.load_state_dict(new_plane_state_dict, strict=False)
    plane_model.to(device)
    if torch.cuda.device_count() > 1:
        plane_model = DataParallel(plane_model)
    plane_model.eval()

    print(f"有效数据: {len(dummy_label_data)} 张图片")

    # 创建切面分类数据集 - 使用带伪标签的数据
    plane_val_pipeline = copy.deepcopy(plane_train_pipeline)
    plane_dataset = Mydataset(dummy_label_data, plane_val_pipeline)
    plane_loader = DataLoader(plane_dataset, shuffle=False,
                              batch_size=plane_data_cfg.get('batch_size', 32),
                              num_workers=plane_data_cfg.get('num_workers', 4),
                              pin_memory=True, collate_fn=collate)

    # 筛选四腔心图片
    four_chamber_paths, four_chamber_targets = filter_four_chamber_images(
        plane_model, plane_loader, device, FOUR_CHAMBER_CLASS_INDEX
    )

    if len(four_chamber_paths) == 0:
        print(f"[警告] 任务 {task_name} 中没有找到四腔心图片，跳过该任务")
        return

    # 创建筛选后的数据集
    filtered_data = create_filtered_dataset(four_chamber_paths, four_chamber_targets)
    print(f"筛选后数据: {len(filtered_data)} 张四腔心图片")

    # ================= 第二阶段：疾病诊断 =================
    print(">>> 第二阶段：加载疾病诊断模型...")
    diagnosis_model_cfg, diagnosis_train_pipeline, _, diagnosis_data_cfg, _, _ = file2dict(diagnosis_config_file)
    diagnosis_classes_names, diagnosis_indexs = get_info(diagnosis_label_map)

    # 加载疾病诊断模型
    diagnosis_model = BuildNet(copy.deepcopy(diagnosis_model_cfg))
    diagnosis_checkpoint = torch.load(diagnosis_checkpoint_file, map_location='cpu')
    diagnosis_state_dict = diagnosis_checkpoint[
        'state_dict'] if 'state_dict' in diagnosis_checkpoint else diagnosis_checkpoint
    new_diagnosis_state_dict = {k.replace('module.', ''): v for k, v in diagnosis_state_dict.items()}
    diagnosis_model.load_state_dict(new_diagnosis_state_dict, strict=False)
    diagnosis_model.to(device)
    if torch.cuda.device_count() > 1:
        diagnosis_model = DataParallel(diagnosis_model)
    diagnosis_model.eval()

    # 创建疾病诊断数据集（仅四腔心图片）
    diagnosis_val_pipeline = copy.deepcopy(diagnosis_train_pipeline)
    diagnosis_dataset = Mydataset(filtered_data, diagnosis_val_pipeline)
    diagnosis_loader = DataLoader(diagnosis_dataset, shuffle=False,
                                  batch_size=diagnosis_data_cfg.get('batch_size', 32),
                                  num_workers=diagnosis_data_cfg.get('num_workers', 4),
                                  pin_memory=True, collate_fn=collate)

    # 疾病诊断推理
        # 疾病诊断推理
    print(">>> 进行疾病诊断推理...")
    preds, targets, image_paths = [], [], []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(diagnosis_loader), total=len(diagnosis_loader)):
            images, target, path = batch

            # ===== 灰度化预处理（tensor层面）=====
            gray = (0.114 * images[:, 0:1, :, :] +
                    0.587 * images[:, 1:2, :, :] +
                    0.299 * images[:, 2:3, :, :])   # → [B, 1, H, W]
            images = gray.expand(-1, 3, -1, -1)     # → [B, 3, H, W]
            # =====================================

            outputs = diagnosis_model(images.to(device), return_loss=False)
            outputs = F.softmax(outputs, dim=1)

            preds.append(outputs.cpu())
            targets.append(target.cpu())
            image_paths.extend(path)

    all_preds = torch.cat(preds)
    all_targets = torch.cat(targets)

    # ================= 第三阶段：病人级别评估 =================
    print(">>> 第三阶段：病人级别评估...")
    metrics_csv = os.path.join(output_dir, 'patient_level_metrics_summary.csv')
    pat_pred_csv = os.path.join(output_dir, 'patient_level_predictions.csv')

    # 病人级别评估
    pat_results, pat_APs, pat_Accs, pat_preds, pat_targets, pat_ids = evaluate_patient_level_by_rules(
        all_preds, all_targets, image_paths, diagnosis_classes_names,
        diagnosis_data_cfg, output_dir, rules=id_rules
    )

    if pat_ids:
        # 只保存病人级别评估结果
        get_metrics_output(pat_results, metrics_csv, diagnosis_classes_names,
                           diagnosis_indexs, pat_APs, pat_Accs, title_prefix="Patient Level")
        get_prediction_output(pat_preds, pat_targets, pat_ids, diagnosis_classes_names,
                              diagnosis_indexs, pat_pred_csv, header_id="Patient_ID")

        # 保存筛选信息
        filter_info_csv = os.path.join(output_dir, 'filter_info.csv')
        with open(filter_info_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['统计信息', '数值'])
            writer.writerow(['原始图片数量', len(image_paths)])
            writer.writerow(['四腔心图片数量', len(four_chamber_paths)])
            writer.writerow(['有效病人数量', len(pat_ids)])
            writer.writerow(['筛选比例(%)', f"{len(four_chamber_paths) / len(image_paths) * 100:.2f}"])

    print(f"任务 [{task_name}] 完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    print(f"=== 全流程批量评测启动 (任务数: {len(BATCH_TASKS)}) ===")
    print(f"流程: 切面分类筛选 → 疾病诊断 → 病人级别评估")
    print(f"统一输出目录: {BASE_OUTPUT_DIR}")

    # 确保基础输出目录存在
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
        print(f"创建输出目录: {BASE_OUTPUT_DIR}")

    for task in BATCH_TASKS:
        try:
            run_single_task(task)
        except Exception as e:
            print(f"\n[Error] 任务 {task['task_name']} 执行失败: {e}")
            import traceback

            traceback.print_exc()
    print("\n所有任务执行完毕。")