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

# ================= 1. 批量评测任务配置 =================

# 公共配置路径 (方便统一修改)
# COMMON_CONFIG = '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/华西诊断训练集增强-重新训练/tinyvit_21m.py'
# COMMON_CHECKPOINT = '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/华西诊断训练集增强-重新训练/Last_Epoch100.pth'
COMMON_LABEL_MAP = '/home/UserData/les/Awesome-Backbones-main/datas/胎儿分类-12.13/class_map.txt'

COMMON_CONFIG = '/home/UserData/les/Awesome-Backbones-main/datas/eval_results/2025-12-04-15-32-00/tinyvit_21m.py'
COMMON_CHECKPOINT = '/home/UserData/zyh/华西/Awesome-Backbones-main/logs/TinyViT/2026-01-19-15-53-49/Val_Epoch145-Acc96.784.pth'

BATCH_TASKS = [
    # --- 任务 1: 德阳增强 ---
    {
        'task_name': '德阳增强',
        'config_file': COMMON_CONFIG,
        'checkpoint_file': COMMON_CHECKPOINT,
        'label_map': COMMON_LABEL_MAP,
        'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/德阳增强-labels.txt',
        'output_dir': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿分类-1.23/评估结果_德阳增强',
        'id_rules': {
            '异常组-52例': 3,
            '正常组-132例（含32例点状强回声）': 2,
        }
    },

    # # --- 任务 2: 德阳未增强 ---
    # {
    #     'task_name': '德阳未增强',
    #     'config_file': COMMON_CONFIG,
    #     'checkpoint_file': COMMON_CHECKPOINT,
    #     'label_map': COMMON_LABEL_MAP,
    #     'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/德阳未增强-labels.txt',
    #     'output_dir': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿分类-1.23/评估结果_德阳未增强',
    #     'id_rules': {
    #         '异常组-52例': 3,
    #         '正常组-132例（含32例点状强回声）': 2,
    #     }
    # },

    # # --- 任务 3: 万源增强 ---
    {
        'task_name': '万源增强',
        'config_file': COMMON_CONFIG,
        'checkpoint_file': COMMON_CHECKPOINT,
        'label_map': COMMON_LABEL_MAP,
        'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/万源增强-labels.txt',
        'output_dir': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿分类-1.23/评估结果_万源增强',
        'id_rules': {
            '2025.3-4月标注-万源正常组-已标注121例-2025.4.19': 2,
            '2025.3.26重新标注-万源市中心医院异常组-专家返回后-11例': 1,
        }
    },

    # # --- 任务 4: 万源未增强 ---
    # {
    #     'task_name': '万源未增强',
    #     'config_file': COMMON_CONFIG,
    #     'checkpoint_file': COMMON_CHECKPOINT,
    #     'label_map': COMMON_LABEL_MAP,
    #     'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/万源未增强-labels.txt',
    #     'output_dir': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿分类-1.23/评估结果_万源未增强',
    #     'id_rules': {
    #         '2025.3-4月标注-万源正常组-已标注121例-2025.4.19': 2,
    #         '2025.3.26重新标注-万源市中心医院异常组-专家返回后-11例': 1,
    #     }
    # },
    # #
    # # --- 任务 5: 省妇保增强 ---
    {
        'task_name': '省妇保增强',
        'config_file': COMMON_CONFIG,
        'checkpoint_file': COMMON_CHECKPOINT,
        'label_map': COMMON_LABEL_MAP,
        'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/省妇保增强-labels.txt',
        'output_dir': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿分类-1.23/评估结果_省妇保增强',
        'id_rules': {
            '2025.5.27叶璐核实完成切面分类标注-20255.15潘仁咪-2025.3.25再次标注-省妇保正常组--375例': 2,
            '2025.6.2叶璐完成切面分类及四腔心图片诊断分类标注-2025.5.27潘仁咪-2025.3.25再次标注完成-2月刘杨陈琳返回-省妇保异常组-375例': 2,
        }
    },

    # # --- 任务 6: 省妇保未增强 ---
    # {
    #     'task_name': '省妇保未增强',
    #     'config_file': COMMON_CONFIG,
    #     'checkpoint_file': COMMON_CHECKPOINT,
    #     'label_map': COMMON_LABEL_MAP,
    #     'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/省妇保未增强-labels.txt',
    #     'output_dir': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿分类-1.23/评估结果_省妇保未增强',
    #     'id_rules': {
    #         '2025.5.27叶璐核实完成切面分类标注-20255.15潘仁咪-2025.3.25再次标注-省妇保正常组--375例': 2,
    #         '2025.6.2叶璐完成切面分类及四腔心图片诊断分类标注-2025.5.27潘仁咪-2025.3.25再次标注完成-2月刘杨陈琳返回-省妇保异常组-375例': 2,
    #     }
    # },
    # {
    #     'task_name': '测试集2未增强',
    #     'config_file': COMMON_CONFIG,
    #     'checkpoint_file': COMMON_CHECKPOINT,
    #     'label_map': COMMON_LABEL_MAP,
    #     'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿分类-12.13/测试集2未增强-labels.txt',
    #     'output_dir': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿分类-1.23/评估结果_测试集2未增强',
    #     'id_rules': {
    #         '2025.3.31再次标注测试集2正常组-全部预处理后-40例': (2,0),
    #         '2025.4.14再次标注测试集2异常组-四腔心图片标注正常与异常-全部预处理后-39例': (2,1),
    #     }
    # },
    {
        'task_name': '测试集2增强',
        'config_file': COMMON_CONFIG,
        'checkpoint_file': COMMON_CHECKPOINT,
        'label_map': COMMON_LABEL_MAP,
        'test_annotations': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿分类-12.13/测试集2增强-labels.txt',
        'output_dir': '/home/UserData/les/Awesome-Backbones-main/datas/胎儿分类-1.23/评估结果_测试集2增强',
        'id_rules': {
            '2025.3.31再次标注测试集2正常组-全部预处理后-40例': (2,0),
            '2025.4.14再次标注测试集2异常组-四腔心图片标注正常与异常-全部预处理后-39例': (2,1),
        }
    },
]


# ================= 2. 核心逻辑修改区域 =================

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
    eval_results = evaluate(final_preds_tensor, final_targets_tensor,
                            data_cfg.get('test').get('metrics'),
                            data_cfg.get('test').get('metric_options'))

    # 4. 绘图
    APs, Accs = plot_ROC_curve(final_preds_tensor, final_targets_tensor, classes_names, save_dir, prefix='Patient_')

    return eval_results, APs, Accs, final_preds_tensor, final_targets_tensor, final_patient_ids


# ================= 3. 辅助函数区域 (保持不变) =================

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
         '{:.2f}'.format(mean(eval_results.get('f1_score', 0.0)))),
    )
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


def run_single_task(task_conf):
    task_name = task_conf['task_name']
    print(f"\n{'=' * 20} 开始任务: {task_name} {'=' * 20}")

    config_file = task_conf['config_file']
    checkpoint_file = task_conf['checkpoint_file']
    label_map = task_conf['label_map']
    test_annotations = task_conf['test_annotations']
    output_dir = task_conf['output_dir']
    id_rules = task_conf['id_rules']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_cfg, train_pipeline, val_pipeline, data_cfg, _, _ = file2dict(config_file)
    classes_names, indexs = get_info(label_map)

    with open(test_annotations, encoding='utf-8') as f:
        test_datas = f.readlines()

    print(f"输入文件: {test_annotations} ({len(test_datas)} 行)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================= [修复点在这里] =================
    # 使用 copy.deepcopy() 传递副本，防止原字典被 pop() 破坏
    import copy
    print("正在构建模型...")
    model = BuildNet(copy.deepcopy(model_cfg))
    # =================================================

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.eval()

    val_pipeline = copy.deepcopy(train_pipeline)
    test_dataset = Mydataset(test_datas, val_pipeline)
    test_loader = DataLoader(test_dataset, shuffle=False,
                             batch_size=data_cfg.get('batch_size'),
                             num_workers=data_cfg.get('num_workers'),
                             pin_memory=True, collate_fn=collate)

    print(">>> 正在进行 Image Level 推理...")
    preds, targets, image_paths = [], [], []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            images, target, path = batch
            outputs = model(images.to(device), return_loss=False)

            # =================== 【修改开始】 ===================
            # 强制切片：如果模型输出通道数 (3) 大于 标签名称数量 (2)
            # 我们直接丢弃多余的通道 (第3列)，只保留前两列
            if outputs.shape[1] > len(classes_names):
                outputs = outputs[:, :len(classes_names)]
            # =================== 【修改结束】 ===================

            outputs = F.softmax(outputs, dim=1)

            preds.append(outputs.cpu())
            targets.append(target.cpu())
            image_paths.extend(path)

    all_preds = torch.cat(preds)
    all_targets = torch.cat(targets)

    metrics_csv = os.path.join(output_dir, 'metrics_summary.csv')
    img_pred_csv = os.path.join(output_dir, 'image_level_predictions.csv')
    pat_pred_csv = os.path.join(output_dir, 'patient_level_predictions.csv')

    # Image Level 评估
    img_eval_results = evaluate(all_preds, all_targets, data_cfg.get('test').get('metrics'),
                                data_cfg.get('test').get('metric_options'))
    img_APs, img_Accs = plot_ROC_curve(all_preds, all_targets, classes_names, output_dir, prefix='Image_')

    get_metrics_output(img_eval_results, metrics_csv, classes_names, indexs, img_APs, img_Accs,
                       title_prefix="Image Level")
    get_prediction_output(all_preds, all_targets, image_paths, classes_names, indexs, img_pred_csv,
                          header_id="File_Path")

    # Patient Level 评估 (根据规则Key定标签)
    print(">>> 正在进行 Patient Level 评估...")
    pat_results, pat_APs, pat_Accs, pat_preds, pat_targets, pat_ids = evaluate_patient_level_by_rules(
        all_preds, all_targets, image_paths, classes_names, data_cfg, output_dir, rules=id_rules
    )

    if pat_ids:
        get_metrics_output(pat_results, metrics_csv, classes_names, indexs, pat_APs, pat_Accs,
                           title_prefix="Patient Level")
        get_prediction_output(pat_preds, pat_targets, pat_ids, classes_names, indexs, pat_pred_csv,
                              header_id="Patient_ID")

    print(f"任务 [{task_name}] 完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    print(f"=== 批量评测启动 (任务数: {len(BATCH_TASKS)}) ===")
    for task in BATCH_TASKS:
        try:
            run_single_task(task)
        except Exception as e:
            print(f"\n[Error] 任务 {task['task_name']} 执行失败: {e}")
            import traceback

            traceback.print_exc()
    print("\n所有任务执行完毕。")