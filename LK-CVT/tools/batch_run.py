# -*- coding: utf-8 -*-
import os
import sys
import copy  # 引入 copy 模块

# 将当前目录添加到环境变量
sys.path.insert(0, os.getcwd())

import evaluation


def run_batch_evaluation():
    # ===================== 用户配置区域 =====================

    # 1. 通用配置
    base_config = {
        'config_file_path': '/home/UserData/les/Awesome-Backbones-main/datas/train_output/测试集1-增强_诊断二分类/tinyvit_21m.py',
        'checkpoint_file_path': '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/华西诊断训练集增强-重新训练/Last_Epoch100.pth',
        'classes_map': '/home/UserData/les/Awesome-Backbones-main/datas/测试集2/class_map.txt',
        'gpu_id': 0
    }

    # 2. 六个测试集的标签文件列表
    test_files_list = [
        # '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/万源未增强-labels.txt',
        # '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/万源增强-labels.txt',
        # '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/德阳未增强-labels.txt',
        # '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/德阳增强-labels.txt',
        # '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/省妇保未增强-labels.txt',
        '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/省妇保增强-labels.txt',
        "/home/UserData/les/Awesome-Backbones-main/datas/测试集2/测试集2增强-labels.txt",
        # "/home/UserData/les/Awesome-Backbones-main/datas/测试集2/测试集2未增强-labels.txt"
    ]

    # 3. 输出根目录
    output_base_dir = '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/华西训练集1重新训练-评估结果'

    # =======================================================

    print(f"准备开始执行 {len(test_files_list)} 个测试任务...\n")

    for i, annotation_path in enumerate(test_files_list):
        if not os.path.exists(annotation_path):
            print(f"[错误] 找不到标签文件: {annotation_path}，跳过。")
            continue

        filename = os.path.basename(annotation_path)
        name_core = filename.replace('-labels.txt', '').replace('.txt', '')

        dir_name = f"诊断分类-{name_core}"
        save_dir = os.path.join(output_base_dir, dir_name)

        print(f"=== 正在运行任务 [{i + 1}/{len(test_files_list)}] ===")
        print(f"Input:  {annotation_path}")
        print(f"Output: {save_dir}")

        try:
            run_single_evaluation(base_config, annotation_path, save_dir)
            print(f"=== 任务 [{i + 1}] 完成 ===\n")
        except Exception as e:
            print(f"=== 任务 [{i + 1}] 失败: {e} ===\n")
            import traceback
            traceback.print_exc()


def run_single_evaluation(config, test_annotation_path, save_dir):
    # 引用函数
    file2dict = evaluation.file2dict
    get_info = evaluation.get_info
    BuildNet = evaluation.BuildNet
    torch = evaluation.torch
    DataParallel = evaluation.DataParallel
    Mydataset = evaluation.Mydataset
    DataLoader = evaluation.DataLoader
    collate = evaluation.collate
    tqdm = evaluation.tqdm
    evaluate = evaluation.evaluate
    plot_ROC_curve = evaluation.plot_ROC_curve
    get_metrics_output = evaluation.get_metrics_output
    get_prediction_output = evaluation.get_prediction_output

    # 路径配置
    config_file_path = config['config_file_path']
    checkpoint_file_path = config['checkpoint_file_path']
    classes_map = config['classes_map']
    gpu_id = config['gpu_id']

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建输出目录
    metrics_output = os.path.join(save_dir, 'metrics_output.csv')
    prediction_output = os.path.join(save_dir, 'prediction_results.csv')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载配置
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(config_file_path)

    # 【修复重点】：深拷贝配置，防止 pop 操作破坏缓存
    model_cfg = copy.deepcopy(model_cfg)

    # 获取类别和数据
    classes_names, indexs = get_info(classes_map)
    with open(test_annotation_path, encoding='utf-8') as f:
        test_datas = f.readlines()

    # 构建模型
    device = torch.device(device_name)
    model = BuildNet(model_cfg)

    # 加载权重
    # print(f"Loading checkpoint: {checkpoint_file_path}")
    checkpoint = torch.load(checkpoint_file_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)

    # 初始化
    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[gpu_id])

    model.to(device)
    model.eval()

    # 数据加载
    val_pipeline = copy.deepcopy(train_pipeline)
    test_dataset = Mydataset(test_datas, val_pipeline)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'),
                             num_workers=data_cfg.get('num_workers'), pin_memory=True, collate_fn=collate)

    # 推理
    with torch.no_grad():
        preds, targets, image_paths = [], [], []
        desc = os.path.basename(save_dir)
        with tqdm(total=len(test_loader), desc=desc) as pbar:
            for _, batch in enumerate(test_loader):
                images, target, image_path = batch
                outputs = model(images.to(device), return_loss=False)
                preds.append(outputs)
                targets.append(target.to(device))
                image_paths.extend(image_path)
                pbar.update(1)

    # 评估与保存
    eval_results = evaluate(torch.cat(preds), torch.cat(targets), data_cfg.get('test').get('metrics'),
                            data_cfg.get('test').get('metric_options'))

    APs = plot_ROC_curve(torch.cat(preds), torch.cat(targets), classes_names, save_dir)
    get_metrics_output(eval_results, metrics_output, classes_names, indexs, APs)
    get_prediction_output(torch.cat(preds), torch.cat(targets), image_paths, classes_names, indexs, prediction_output)


if __name__ == "__main__":
    run_batch_evaluation()